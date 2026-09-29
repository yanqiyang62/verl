# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio

from verl.trainer.ppo.v1.agent_loop_tq import _settle_session_tasks


def test_settle_session_tasks_waits_for_siblings_after_failure():
    async def run():
        settled = asyncio.Event()

        async def fail():
            raise RuntimeError("session failed")

        async def finish_later():
            await asyncio.sleep(0.01)
            settled.set()

        tasks = [asyncio.create_task(fail()), asyncio.create_task(finish_later())]
        errors = await _settle_session_tasks(tasks)

        assert settled.is_set()
        assert all(task.done() for task in tasks)
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)

    asyncio.run(run())


def test_deadline_settles_cancelled_trajectory_before_group_terminal():
    from verl.trainer.ppo.v1.agent_loop_tq import _run_session_with_deadline

    async def run():
        writes = []
        cleanup = asyncio.Event()

        async def hangs():
            try:
                await asyncio.Event().wait()
                writes.append("late-write")
            finally:
                await asyncio.sleep(0.01)
                cleanup.set()

        tasks = [asyncio.create_task(_run_session_with_deadline(hangs(), 0.01, "slow-group", 0))]
        errors = await _settle_session_tasks(tasks)
        assert cleanup.is_set()
        assert tasks[0].done()
        assert len(errors) == 1
        assert isinstance(errors[0], TimeoutError)
        assert "slow-group" in str(errors[0]) and "sample_id=0" in str(errors[0])
        await asyncio.sleep(0.01)
        assert writes == []

    asyncio.run(run())


def test_fifteen_complete_one_stuck_publishes_three_finished_one_failure(monkeypatch):
    from types import SimpleNamespace

    import verl.trainer.ppo.v1.agent_loop_tq as module

    async def run():
        tags, writes, cancelled = {}, [], []

        async def put(key, partition_id, tag):
            if tag["status"] == "failure":
                assert cancelled == [("group3", 0)]  # No terminal status before cancellation settles.
            tags[key] = tag["status"]

        async def session(params, *, uid, session_id, **kwargs):
            if uid == "group3" and session_id == 0:
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.append((uid, session_id))
            else:
                await asyncio.sleep(0.001)
                writes.append((uid, session_id))

        monkeypatch.setattr(module.tq, "async_kv_put", put)
        worker = SimpleNamespace(
            config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(n=4))),
            trajectory_timeout=0.03,
            _run_agent_loop=session,
        )
        method = module.AgentLoopWorkerTQ.__ray_metadata__.modified_class._run_prompt
        await asyncio.gather(*(method(worker, {"uid": f"group{i}"}, {}, {"validate": False}) for i in range(4)))
        assert len(writes) == 15
        assert list(tags.values()).count("finished") == 3
        assert tags["group3"] == "failure"
        await asyncio.sleep(0.01)
        assert len(writes) == 15

    asyncio.run(run())
