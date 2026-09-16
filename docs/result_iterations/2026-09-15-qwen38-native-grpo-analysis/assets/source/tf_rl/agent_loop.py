"""Generate ONE new assistant decision from a frozen historical prefix.
Uses the model's own tools-aware template. Never executes tools or replays a
teacher target. The exact sampled response IDs are returned for RL updates.
"""
import inspect
import time
from uuid import uuid4
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from tf_rl.rendering import prompt_tokens


@register('fixed_prefix_agent')
class FixedPrefixAgentLoop(AgentLoopBase):
    async def run(self,sampling_params,priority=0,**kwargs):
        info=kwargs.get('extra_info') or {}
        if 'tools_json' not in info:
            raise ValueError('Per-row tools missing; use FixedPrefixDataset and this AgentLoop together')
        messages=kwargs['raw_prompt']
        ids=prompt_tokens(self.tokenizer,messages,info['tools_json'],self.apply_chat_template_kwargs)
        if len(ids)>int(self.rollout_config.prompt_length):
            raise ValueError('Rollout prompt exceeds configured budget; no silent truncation')
        # Stop at the model's normal EOS; do not stop before closing </tool_call>.
        params=dict(sampling_params)
        call={'request_id':uuid4().hex,'prompt_ids':ids,'sampling_params':params}
        if 'priority' in inspect.signature(self.server_manager.generate).parameters:
            call['priority']=int(priority)
        start=time.perf_counter()
        output=await self.server_manager.generate(**call)
        response=list(output.token_ids)
        if not response:raise ValueError('Rollout returned an empty response')
        if len(response)>int(self.rollout_config.response_length):
            raise ValueError('Backend exceeded configured response budget')
        logprobs=getattr(output,'log_probs',None)
        if logprobs is not None and len(logprobs)!=len(response):
            raise ValueError('Response/logprob length mismatch')
        extra=dict(getattr(output,'extra_fields',None) or {})
        extra.update(turn_scores=[],tool_rewards=[])
        # Supported API fields are checked by preflight before training.
        return AgentLoopOutput(prompt_ids=ids,response_ids=response,
            response_mask=[1]*len(response),response_logprobs=logprobs,
            num_turns=1,metrics={'generate_sequences':time.perf_counter()-start},extra_fields=extra)
