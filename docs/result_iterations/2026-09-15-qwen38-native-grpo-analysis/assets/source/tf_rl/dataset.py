"""Tools-aware Parquet adapter for verl's custom_cls hook."""
import copy
import json
import random
import torch
from torch.utils.data import Dataset
import pyarrow.parquet as pq
from tf_rl.rendering import decode_messages, prompt_tokens


class FixedPrefixDataset(Dataset):
    def __init__(self, data_files, tokenizer, config, processor=None, max_samples=-1, **kwargs):
        # verl V1 may pass a non-None processor even for a text-only model/run.
        # This dataset is intentionally text-only and uses tokenizer only, so ignore it.
        processor = None
        self.tokenizer=tokenizer;self.config=config
        self.max_prompt_length=int(config.get('max_prompt_length',8192))
        self.template_kwargs=dict(config.get('apply_chat_template_kwargs',{}))
        files=[data_files] if isinstance(data_files,str) else list(data_files)
        self.rows=[];self.prompt_ids=[];self.dropped=[]
        rows=[row for path in files for row in pq.read_table(path).to_pylist()]
        if max_samples is not None and max_samples>0 and len(rows)>max_samples:
            rows=random.Random(config.get('seed') or 42).sample(rows,max_samples)
        for row in rows:
            info=json.loads(row['extra_info']) if isinstance(row['extra_info'],str) else row['extra_info']
            if info['reward_kind'] not in ('tool_call','json'):
                raise ValueError('Use data/rule_train.parquet; all_train contains text without a semantic judge')
            ids=prompt_tokens(tokenizer,row['prompt'],row['tools'],self.template_kwargs)
            if len(ids)>self.max_prompt_length:
                if not config.get('filter_overlong_prompts',False):
                    raise ValueError(f"Prompt too long at source line {info['source_line']}: {len(ids)}")
                self.dropped.append(info['source_line']);continue
            self.rows.append(row);self.prompt_ids.append(ids)
        if not self.rows:raise ValueError('No eligible data remains')
        print(f'FixedPrefixDataset: kept={len(self.rows)}, overlong_dropped={len(self.dropped)}')

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        row=copy.deepcopy(self.rows[index]);ids=list(self.prompt_ids[index])
        info=json.loads(row['extra_info']) if isinstance(row['extra_info'],str) else row['extra_info']
        # Carry row-specific tool definitions through both legacy and new AgentLoop batching.
        info.update(index=index,tools_json=row['tools'],need_tools_kwargs=False)
        pad_id=self.tokenizer.pad_token_id
        if pad_id is None:pad_id=self.tokenizer.eos_token_id
        if pad_id is None:raise ValueError('Tokenizer has no pad/eos token')
        n=self.max_prompt_length-len(ids)
        mask=[0]*n+[1]*len(ids)
        return {'input_ids':torch.tensor([pad_id]*n+ids,dtype=torch.long),
                'attention_mask':torch.tensor(mask,dtype=torch.long),
                'position_ids':torch.tensor([0]*n+list(range(len(ids))),dtype=torch.long),
                'raw_prompt':decode_messages(row['prompt']),'raw_prompt_ids':ids,
                'reward_model':row['reward_model'],'data_source':row['data_source'],
                'extra_info':info,'index':index,'agent_name':'fixed_prefix_agent',
                'ability':row['ability']}

    def resume_dataset_state(self):
        # Dataset is fully serializable; do not re-split or change ordering on resume.
        return None

    def split(self,num_splits):
        if not isinstance(num_splits,int) or num_splits<1:raise ValueError('num_splits must be positive')
        if num_splits>len(self):raise ValueError('More splits than rows')
        parts=[]
        for i in range(num_splits):
            lo=len(self)*i//num_splits;hi=len(self)*(i+1)//num_splits
            obj=copy.copy(self);obj.rows=self.rows[lo:hi];obj.prompt_ids=self.prompt_ids[lo:hi];parts.append(obj)
        return parts
