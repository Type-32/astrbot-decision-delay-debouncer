import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


def load_plugin():
    modules = {
        name: types.ModuleType(name)
        for name in [
            "astrbot",
            "astrbot.api",
            "astrbot.api.star",
            "astrbot.api.event",
            "astrbot.api.provider",
            "astrbot.api.message_components",
        ]
    }
    api = modules["astrbot.api"]
    api.logger = types.SimpleNamespace(
        **{name: lambda *_a, **_k: None for name in ["debug", "warning", "error"]}
    )
    api.AstrBotConfig = dict
    star = modules["astrbot.api.star"]

    class Star:
        def __init__(self, context):
            self.context = context

    star.Star = Star
    star.Context = object
    event = modules["astrbot.api.event"]
    event.AstrMessageEvent = object

    class CustomFilter:
        def __init__(self, raise_error=True, **_):
            self.raise_error = raise_error

    event.filter = types.SimpleNamespace(
        CustomFilter=CustomFilter,
        PermissionType=types.SimpleNamespace(ADMIN="admin"),
        **{
            name: lambda *_a, **_k: lambda func: func
            for name in [
                "on_astrbot_loaded",
                "on_waiting_llm_request",
                "on_llm_request",
                "on_llm_response",
                "custom_filter",
                "command",
                "permission_type",
            ]
        },
    )
    modules["astrbot.api.provider"].ProviderRequest = object
    modules["astrbot.api.provider"].LLMResponse = object
    modules["astrbot.api.message_components"].Plain = lambda text: types.SimpleNamespace(
        type="Plain", text=text
    )
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "debounce_test_module", root / "main.py", submodule_search_locations=[str(root)]
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module


debounce = load_plugin()


def picture(name):
    return types.SimpleNamespace(type="Image", file=f"base64://{name}")


class Event:
    """The slice of AstrMessageEvent the plugin touches."""

    def __init__(self, text, images=(), sender="u1", private=True, addressed=True):
        self.message_str = text
        plain = [types.SimpleNamespace(type="Plain", text=text)] if text else []
        self.message_obj = types.SimpleNamespace(
            message=plain + list(images), message_str=text
        )
        self.unified_msg_origin = "qq:FriendMessage:1" if private else "qq:Group:9"
        self.sender = sender
        self.private = private
        self.is_at_or_wake_command = addressed
        self.extras = {}
        self.files = []
        self.stopped = False

    def is_private_chat(self):
        return self.private

    def get_sender_id(self):
        return self.sender

    def get_platform_name(self):
        return "aiocqhttp"

    def get_extra(self, key, default=None):
        return self.extras.get(key, default)

    def set_extra(self, key, value):
        self.extras[key] = value

    def stop_event(self):
        self.stopped = True

    def is_stopped(self):
        return self.stopped

    def track_temporary_local_file(self, path):
        self.files.append(path)

    def untrack_temporary_local_file(self, path):
        self.files.remove(path)

    @property
    def _temporary_local_files(self):
        return self.files


class BurstTests(unittest.IsolatedAsyncioTestCase):
    """Delays are shrunk so the real clock runs each case in well under a second."""

    async def asyncSetUp(self):
        self.config = {
            "classifier_backend": "none",
            "delay_seconds": 0.05,
            "max_wait_seconds": 1,
        }
        self.plugin = debounce.DebouncePlugin(None, self.config)
        self.judge = AsyncMock(return_value=1.0)
        self.plugin._judge = self.judge

    async def asyncTearDown(self):
        await self.plugin.terminate()

    def use_judge(self, *scores):
        self.judge.side_effect = list(scores)

    def lead(self, event):
        task = asyncio.create_task(self.plugin.on_waiting_llm_request(event))
        return task

    async def follow(self, event):
        """A later message: early handler first, then the LLM hook if it survives."""
        if self.plugin._follow_up_kind(event):
            await self.plugin.on_follow_up(event)
        if not event.stopped:
            return self.lead(event)

    async def test_rapid_fire_merges_into_leader(self):
        first = Event("Well")
        task = self.lead(first)
        await asyncio.sleep(0.01)
        second = Event("ok", [picture("a")])
        await self.follow(second)
        await task
        self.assertTrue(second.stopped)
        self.assertFalse(first.stopped)
        self.assertEqual(first.message_str, "Well\nok")
        self.assertEqual(first.message_obj.message_str, "Well\nok")
        self.assertEqual(
            [part.type for part in first.message_obj.message], ["Plain", "Image"]
        )

    async def test_single_message_is_untouched(self):
        event = Event("hello", [picture("a")])
        original = event.message_obj.message
        await self.lead(event)
        self.assertFalse(event.stopped)
        self.assertIs(event.message_obj.message, original)

    def test_reading_delay(self):
        delay = debounce.reading_delay
        self.assertEqual(delay("Well \n ok", 0, 0.8), 0.8)
        self.assertEqual(delay(" ".join(["word"] * 12), 0, 0.8), 3.0)
        self.assertEqual(delay(" ".join(["word"] * 40), 0, 0.8), 5.0)
        self.assertEqual(delay("", 1, 0.8), 1.0)
        self.assertAlmostEqual(delay("如果明天不下雨", 0, 0.8), 0.875)
        self.assertEqual(delay("anything long " * 20, 3, 0), 0.0)

    async def test_incomplete_burst_waits_for_next_message(self):
        self.use_judge(0.1, 0.95)
        first = Event("如果明天")
        task = self.lead(first)
        await asyncio.sleep(0.2)  # past the delay: judged incomplete, still held
        self.assertFalse(task.done())
        await self.follow(Event("我们去爬山吧"))
        await task
        self.assertEqual(first.message_str, "如果明天\n我们去爬山吧")
        self.assertEqual(self.judge.await_args.args[0].texts, ["如果明天", "我们去爬山吧"])

    async def test_max_wait_releases_incomplete_burst(self):
        self.use_judge(0.1)
        self.config["max_wait_seconds"] = 0.5  # longer than the 0.25 s read time
        event = Event("hi")
        await asyncio.wait_for(self.lead(event), 1)
        self.assertFalse(event.stopped)
        self.judge.assert_awaited_once()

    async def test_max_wait_beats_reading_delay(self):
        self.config["max_wait_seconds"] = 0.2
        event = Event(" ".join(["word"] * 40))  # 5 s reading delay
        await asyncio.wait_for(self.lead(event), 1)
        self.assertFalse(event.stopped)

    async def test_message_during_judgment_rejudges_merged_text(self):
        started, release = asyncio.Event(), asyncio.Event()

        async def slow(burst):
            if self.judge.await_count == 1:
                started.set()
                await release.wait()
            return 0.99

        self.judge.side_effect = slow
        first = Event("如果明天")
        task = self.lead(first)
        await started.wait()
        await self.follow(Event("不下雨"))
        release.set()
        await task
        self.assertEqual(self.judge.await_count, 2)
        self.assertEqual(first.message_str, "如果明天\n不下雨")

    async def test_stopped_leader_drops_burst(self):
        first = Event("hello")
        self.config["delay_seconds"] = 0.3
        task = self.lead(first)
        await asyncio.sleep(0.01)
        first.set_extra("agent_stop_requested", True)  # /stop while waiting
        await task
        self.assertTrue(first.stopped)
        self.assertEqual(self.plugin.bursts, {})

    async def test_senders_in_a_group_are_separate(self):
        a = Event("我觉得", sender="a", private=False)
        b = Event("你好", sender="b", private=False)
        task_a, task_b = self.lead(a), self.lead(b)
        await asyncio.gather(task_a, task_b)
        self.assertEqual((a.message_str, b.message_str), ("我觉得", "你好"))
        self.assertFalse(a.stopped or b.stopped)

    async def test_group_followups_without_at_need_opt_in(self):
        leader = Event("我想问", sender="a", private=False)
        task = self.lead(leader)
        await asyncio.sleep(0.01)
        stray = Event("顺便说一下", sender="a", private=False, addressed=False)
        self.assertIsNone(self.plugin._follow_up_kind(stray))
        self.config["merge_group_followups"] = True
        await self.follow(stray)
        await task
        self.assertTrue(stray.stopped)
        self.assertEqual(leader.message_str, "我想问\n顺便说一下")

    async def test_commands_are_not_absorbed(self):
        leader = Event("hello")
        task = self.lead(leader)
        await asyncio.sleep(0.01)
        command = Event("/reset")
        command.set_extra("handlers_parsed_params", {"reset": {}})
        await self.plugin.on_follow_up(command)
        self.assertFalse(command.stopped)
        await task

    async def test_absorbed_media_survives_follower_cleanup(self):
        leader = Event("看这个")
        task = self.lead(leader)
        await asyncio.sleep(0.01)
        follower = Event("", [picture("a")])
        follower.files.append("/tmp/a.jpg")
        await self.follow(follower)
        await task
        self.assertEqual((follower.files, leader.files), ([], ["/tmp/a.jpg"]))

    async def run_reply(self):
        """Release a leader, then hold its pipeline task open like a running LLM call."""
        leader = Event("帮我写首诗")
        req = types.SimpleNamespace(conversation="conv")
        replying = asyncio.Event()

        async def pipeline():
            await self.plugin.on_waiting_llm_request(leader)
            await self.plugin.on_llm_request(leader, req)
            await replying.wait()

        task = asyncio.create_task(pipeline())
        while self.plugin.runs.get(self.plugin._key(leader)) is None:
            await asyncio.sleep(0.01)
        return leader, req, replying, task

    async def test_cancel_aborts_reply_and_carries_text(self):
        leader, req, replying, task = await self.run_reply()
        newcomer = Event("关于秋天")
        lead_task = await self.follow(newcomer)
        self.assertTrue(leader.get_extra("agent_stop_requested"))
        self.assertIsNone(req.conversation)  # aborted turn stays out of history
        self.assertFalse(newcomer.stopped)
        replying.set()
        await asyncio.gather(task, lead_task)
        self.assertEqual(newcomer.message_str, "帮我写首诗\n关于秋天")

    async def test_cancel_disabled_leaves_reply_alone(self):
        self.config["cancel_on_new_message"] = False
        leader, req, replying, task = await self.run_reply()
        newcomer = Event("关于秋天")
        lead_task = await self.follow(newcomer)
        replying.set()
        await asyncio.gather(task, lead_task)
        self.assertIsNone(leader.get_extra("agent_stop_requested"))
        self.assertEqual(req.conversation, "conv")
        self.assertEqual(newcomer.message_str, "关于秋天")

    async def test_answered_or_finished_reply_is_not_cancelled(self):
        leader, req, replying, task = await self.run_reply()
        await self.plugin.on_llm_response(leader, None)
        self.assertIsNone(self.plugin._follow_up_kind(Event("谢谢")))
        replying.set()
        await task
        self.assertEqual(self.plugin.runs, {})

    async def test_unload_releases_waiting_leader(self):
        self.config["delay_seconds"] = 5
        event = Event("hello")
        task = self.lead(event)
        await asyncio.sleep(0.01)
        await self.plugin.terminate()
        await asyncio.wait_for(task, 1)
        self.assertFalse(event.stopped)

    async def test_disabled_plugin_passes_through(self):
        self.config["enabled"] = False
        self.config["delay_seconds"] = 5
        event = Event("hello")
        await asyncio.wait_for(self.lead(event), 0.5)
        self.assertEqual(self.plugin.bursts, {})


class JevJudgeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.plugin = debounce.DebouncePlugin(
            None, {"jev_api_key": "test-only-key", "delay_seconds": 0}
        )
        self.evaluate = self.plugin.jev.evaluate = AsyncMock()

    async def asyncTearDown(self):
        await self.plugin.terminate()

    async def judge(self, text):
        event = Event(text)
        await asyncio.wait_for(self.plugin.on_waiting_llm_request(event), 1)
        return event

    def test_defaults_to_openrouter(self):
        client = self.plugin.jev
        self.assertEqual(client.endpoint, "https://openrouter.ai/api/v1/systemone")
        self.assertEqual(client.model, "jev-latest")
        self.assertTrue(client.inline_images)

    async def test_images_from_all_fragments_reach_judge(self):
        first, second = picture("first"), picture("second")
        self.evaluate.side_effect = [
            {"complete": {"noul": 0.1}},
            {"complete": {"noul": 0.99}},
        ]
        leader = Event("比较这两张图", [first])
        task = asyncio.create_task(self.plugin.on_waiting_llm_request(leader))
        while not self.evaluate.await_count:
            await asyncio.sleep(0.01)
        follower = Event("有什么不同？", [second])
        await self.plugin.on_follow_up(follower)
        await task
        self.assertEqual(self.evaluate.call_args.kwargs["images"], [first, second])
        state, questions = self.evaluate.call_args.args
        self.assertEqual(state["message"], "比较这两张图\n有什么不同？")
        self.assertEqual(questions["complete"]["type"], "noul")
        self.assertEqual(
            [part.type for part in leader.message_obj.message],
            ["Plain", "Image", "Image"],
        )

    async def test_service_failure_sends(self):
        self.evaluate.side_effect = ValueError("JEV HTTP 500")
        self.assertFalse((await self.judge("hello")).stopped)

    async def test_bad_probability_sends(self):
        self.evaluate.return_value = {"complete": {"noul": True}}
        self.assertFalse((await self.judge("hello")).stopped)

    async def test_missing_key_skips_judging(self):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": ""}):
            plugin = debounce.DebouncePlugin(None, {"delay_seconds": 0})
        plugin.jev.evaluate = AsyncMock()
        event = Event("如果")
        await asyncio.wait_for(plugin.on_waiting_llm_request(event), 1)
        self.assertFalse(event.stopped)
        plugin.jev.evaluate.assert_not_awaited()
        await plugin.terminate()

    async def command_reply(self):
        event = Event("/debounce_test")
        event.plain_result = lambda text: text
        return [reply async for reply in self.plugin.debounce_test(event)][0]

    async def test_connection_test_reports_success_and_failure(self):
        self.evaluate.return_value = {"complete": {"type": "noul", "noul": 0.97}}
        self.assertIn("✅ 连接正常，完整概率 0.97", await self.command_reply())
        self.evaluate.side_effect = ValueError("JEV HTTP 403: Key limit exceeded")
        reply = await self.command_reply()
        self.assertIn("❌ 失败：ValueError: JEV HTTP 403: Key limit exceeded", reply)
        self.assertIn("https://openrouter.ai/api/v1/systemone", reply)
        self.assertNotIn("test-only-key", reply)


if __name__ == "__main__":
    unittest.main()
