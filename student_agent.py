import heapq
import re
from itertools import count

class AssemblyAgent:
    def __init__(self):
        self.system_prompt_objects = "You are a deterministic symbolic planner. Return only actions."
        self.system_prompt_blocks = "You are a deterministic blocks-world planner. Return only actions."

        self._obj_patterns = [
            (r"\(\s*attack\s+([a-z])\s*\)", lambda m: f"(attack {m.group(1).lower()})"),
            (r"\(\s*succumb\s+([a-z])\s*\)", lambda m: f"(succumb {m.group(1).lower()})"),
            (r"\(\s*feast\s+([a-z])\s+([a-z])\s*\)", lambda m: f"(feast {m.group(1).lower()} {m.group(2).lower()})"),
            (r"\(\s*overcome\s+([a-z])\s+([a-z])\s*\)", lambda m: f"(overcome {m.group(1).lower()} {m.group(2).lower()})"),
            (r"\battack\s+(?:object\s+)?([a-z])\b", lambda m: f"(attack {m.group(1).lower()})"),
            (r"\bsuccumb\s+(?:object\s+)?([a-z])\b", lambda m: f"(succumb {m.group(1).lower()})"),
            (r"\bfeast\s+(?:object\s+)?([a-z])\s+(?:from\s+(?:object\s+)?)?([a-z])\b", lambda m: f"(feast {m.group(1).lower()} {m.group(2).lower()})"),
            (r"\bovercome\s+(?:object\s+)?([a-z])\s+(?:from\s+(?:object\s+)?)?([a-z])\b", lambda m: f"(overcome {m.group(1).lower()} {m.group(2).lower()})"),
        ]

        self._blk_patterns = [
            (r"\(\s*engage_payload\s+([a-z]+)\s*\)", lambda m: f"(engage_payload {m.group(1).lower()})"),
            (r"\(\s*release_payload\s+([a-z]+)\s*\)", lambda m: f"(release_payload {m.group(1).lower()})"),
            (r"\(\s*unmount_node\s+([a-z]+)\s+([a-z]+)\s*\)", lambda m: f"(unmount_node {m.group(1).lower()} {m.group(2).lower()})"),
            (r"\(\s*mount_node\s+([a-z]+)\s+([a-z]+)\s*\)", lambda m: f"(mount_node {m.group(1).lower()} {m.group(2).lower()})"),
            (r"\bpick\s*up\s+(?:the\s+)?([a-z]+)\s+block\b", lambda m: f"(engage_payload {m.group(1).lower()})"),
            (r"\bput\s*down\s+(?:the\s+)?([a-z]+)\s+block\b", lambda m: f"(release_payload {m.group(1).lower()})"),
            (r"\b(?:unmount_node|unstack)\s+(?:the\s+)?([a-z]+)\s+block\s+from\s+on\s+top\s+of\s+(?:the\s+)?([a-z]+)\s+block\b", lambda m: f"(unmount_node {m.group(1).lower()} {m.group(2).lower()})"),
            (r"\b(?:mount_node|stack)\s+(?:the\s+)?([a-z]+)\s+block\s+on\s+top\s+of\s+(?:the\s+)?([a-z]+)\s+block\b", lambda m: f"(mount_node {m.group(1).lower()} {m.group(2).lower()})"),
        ]

    def _split_facts(self, text: str) -> list[str]:
        return [piece.strip() for piece in re.split(r",|\band\b", text.strip().lower().rstrip(".")) if piece.strip()]

    def _extract_final_statement(self, scenario_context: str) -> str:
        parts = scenario_context.split("[STATEMENT]")
        return parts[-1] if len(parts) > 1 else scenario_context

    def _extract_initial_goal(self, scenario_context: str):
        statement = self._extract_final_statement(scenario_context)
        lines = [line.strip() for line in statement.splitlines() if line.strip()]
        initial_line, goal_line = "", ""

        for line in lines:
            lower = line.lower()
            if lower.startswith("as initial conditions i have that"):
                initial_line = line
            elif lower.startswith("my goal is to have that"):
                goal_line = line

        if not initial_line or not goal_line:
            return None, None

        initial_text = re.sub(r"^as initial conditions i have that,?\s*", "", initial_line, flags=re.IGNORECASE)
        goal_text = re.sub(r"^my goal is to have that,?\s*", "", goal_line, flags=re.IGNORECASE)
        return initial_text, goal_text

    def _normalize_blocks(self, names) -> set[str]:
        banned = {"other"}
        return {name for name in names if name and name not in banned}

    def _extract_blocks_from_statement(self, statement_text: str) -> set[str]:
        return self._normalize_blocks(re.findall(r"the\s+([a-z]+)\s+block", statement_text.lower()))

    def _best_first(self, s0, key_fn, goal_fn, succ_fn, h_fn, max_depth=30, max_exp=90000):
        frontier = []
        tie = count()
        heapq.heappush(frontier, (h_fn(s0), 0, next(tie), s0, []))
        best_g = {key_fn(s0): 0}
        expansions = 0

        while frontier and expansions < max_exp:
            _, g, _, state, plan = heapq.heappop(frontier)
            key = key_fn(state)
            if g > best_g.get(key, 10**9):
                continue
            if goal_fn(state):
                return plan
            if g >= max_depth:
                continue

            expansions += 1
            for action, next_state in succ_fn(state):
                next_g = g + 1
                next_key = key_fn(next_state)
                if next_g >= best_g.get(next_key, 10**9):
                    continue
                best_g[next_key] = next_g
                heapq.heappush(frontier, (next_g + h_fn(next_state), next_g, next(tie), next_state, plan + [action]))

        return []

    def _trim_dup(self, actions: list[str]) -> list[str]:
        n = len(actions)
        if n >= 2 and n % 2 == 0 and actions[: n // 2] == actions[n // 2:]:
            return actions[: n // 2]
        return actions

    def _extract_actions(self, text: str, patterns):
        candidates = []
        for line in text.splitlines():
            normalized_line = line.strip().lower().replace("-", " ")
            for pattern, fmt in patterns:
                match = re.search(pattern, normalized_line, re.IGNORECASE)
                if match:
                    candidates.append(fmt(match))
                    break
        if candidates:
            return self._trim_dup(candidates)

        spans = []
        normalized_text = text.lower().replace("-", " ")
        for pattern, fmt in patterns:
            for match in re.finditer(pattern, normalized_text, re.IGNORECASE):
                spans.append((match.start(), fmt(match)))
        spans.sort(key=lambda item: item[0])
        return self._trim_dup([action for _, action in spans])

    def _sanitize_block_actions(self, actions: list[str], blocks: set[str]) -> list[str]:
        sanitized = []
        for action in actions:
            m = re.fullmatch(r"\(engage_payload\s+([a-z]+)\)", action)
            if m:
                x = m.group(1)
                if x in blocks:
                    sanitized.append(f"(engage_payload {x})")
                continue

            m = re.fullmatch(r"\(release_payload\s+([a-z]+)\)", action)
            if m:
                x = m.group(1)
                if x in blocks:
                    sanitized.append(f"(release_payload {x})")
                continue

            m = re.fullmatch(r"\(unmount_node\s+([a-z]+)\s+([a-z]+)\)", action)
            if m:
                x, y = m.group(1), m.group(2)
                if x in blocks and y in blocks and x != y:
                    sanitized.append(f"(unmount_node {x} {y})")
                continue

            m = re.fullmatch(r"\(mount_node\s+([a-z]+)\s+([a-z]+)\)", action)
            if m:
                x, y = m.group(1), m.group(2)
                if x in blocks and y in blocks and x != y:
                    sanitized.append(f"(mount_node {x} {y})")
                continue

        return self._trim_dup(sanitized)

    # ---------- objects domain ----------
    def _parse_objects(self, scenario_context: str):
        initial_text, goal_text = self._extract_initial_goal(scenario_context)
        if not initial_text or not goal_text:
            return None

        init_facts = self._split_facts(initial_text)
        goal_facts = self._split_facts(goal_text)
        objects = set(re.findall(r"object\s+([a-z])", scenario_context.lower()))

        state = {"harmony": False, "planet": set(), "province": set(), "pain": set(), "craves": set()}
        goal = {"craves": set()}

        for fact in init_facts:
            m = re.fullmatch(r"object\s+([a-z])\s+craves\s+object\s+([a-z])", fact)
            if m:
                x, y = m.group(1), m.group(2)
                state["craves"].add((x, y))
                objects.update([x, y])
                continue
            m = re.fullmatch(r"planet\s+object\s+([a-z])", fact)
            if m:
                x = m.group(1)
                state["planet"].add(x)
                objects.add(x)
                continue
            m = re.fullmatch(r"province\s+object\s+([a-z])", fact)
            if m:
                x = m.group(1)
                state["province"].add(x)
                objects.add(x)
                continue
            if fact == "harmony":
                state["harmony"] = True

        for fact in goal_facts:
            m = re.fullmatch(r"object\s+([a-z])\s+craves\s+object\s+([a-z])", fact)
            if m:
                x, y = m.group(1), m.group(2)
                goal["craves"].add((x, y))
                objects.update([x, y])

        return state, goal, sorted(objects)

    def _obj_key(self, state):
        return (
            state["harmony"],
            tuple(sorted(state["planet"])),
            tuple(sorted(state["province"])),
            tuple(sorted(state["pain"])),
            tuple(sorted(state["craves"])),
        )

    def _obj_goal(self, state, goal):
        return goal["craves"].issubset(state["craves"])

    def _obj_copy(self, state):
        return {
            "harmony": state["harmony"],
            "planet": set(state["planet"]),
            "province": set(state["province"]),
            "pain": set(state["pain"]),
            "craves": set(state["craves"]),
        }

    def _obj_succ(self, state, objects):
        out = []

        for x in objects:
            for y in objects:
                if x != y and state["harmony"] and x in state["province"] and (x, y) in state["craves"]:
                    ns = self._obj_copy(state)
                    ns["pain"].add(x)
                    ns["province"].add(y)
                    ns["craves"].discard((x, y))
                    ns["province"].discard(x)
                    ns["harmony"] = False
                    out.append((f"(feast {x} {y})", ns))

        for x in objects:
            if state["harmony"] and x in state["planet"] and x in state["province"]:
                ns = self._obj_copy(state)
                ns["pain"].add(x)
                ns["province"].discard(x)
                ns["planet"].discard(x)
                ns["harmony"] = False
                out.append((f"(attack {x})", ns))

        for x in objects:
            if x in state["pain"]:
                ns = self._obj_copy(state)
                ns["province"].add(x)
                ns["planet"].add(x)
                ns["harmony"] = True
                ns["pain"].discard(x)
                out.append((f"(succumb {x})", ns))

        for x in objects:
            for y in objects:
                if x != y and y in state["province"] and x in state["pain"]:
                    ns = self._obj_copy(state)
                    ns["harmony"] = True
                    ns["province"].add(x)
                    ns["craves"].add((x, y))
                    ns["province"].discard(y)
                    ns["pain"].discard(x)
                    out.append((f"(overcome {x} {y})", ns))

        return out

    def _plan_objects(self, scenario_context: str) -> list[str]:
        parsed = self._parse_objects(scenario_context)
        if parsed is None:
            return []
        state0, goal, objects = parsed
        if self._obj_goal(state0, goal):
            return []

        return self._best_first(
            state0,
            key_fn=self._obj_key,
            goal_fn=lambda s: self._obj_goal(s, goal),
            succ_fn=lambda s: self._obj_succ(s, objects),
            h_fn=lambda s: 2 * len(goal["craves"] - s["craves"]),
            max_depth=max(16, 5 * max(1, len(goal["craves"]))),
            max_exp=90000,
        )

    # ---------- blocks domain ----------
    def _parse_blocks(self, scenario_context: str):
        initial_text, goal_text = self._extract_initial_goal(scenario_context)
        if not initial_text or not goal_text:
            return None

        init_facts = self._split_facts(initial_text)
        goal_facts = self._split_facts(goal_text)
        blocks = set()

        state = {"on": set(), "ontable": set(), "holding": None}
        goal = {"on": set(), "ontable": set()}

        for fact in init_facts:
            m = re.fullmatch(r"the\s+([a-z]+)\s+block\s+is\s+on\s+top\s+of\s+the\s+([a-z]+)\s+block", fact)
            if m:
                x, y = m.group(1), m.group(2)
                state["on"].add((x, y))
                blocks.update([x, y])
                continue
            m = re.fullmatch(r"the\s+([a-z]+)\s+block\s+is\s+on\s+the\s+table", fact)
            if m:
                x = m.group(1)
                state["ontable"].add(x)
                blocks.add(x)
                continue
            m = re.fullmatch(r"i\s+am\s+holding\s+the\s+([a-z]+)\s+block", fact)
            if m:
                x = m.group(1)
                state["holding"] = x
                blocks.add(x)

        for fact in goal_facts:
            m = re.fullmatch(r"the\s+([a-z]+)\s+block\s+is\s+on\s+top\s+of\s+the\s+([a-z]+)\s+block", fact)
            if m:
                x, y = m.group(1), m.group(2)
                goal["on"].add((x, y))
                blocks.update([x, y])
                continue
            m = re.fullmatch(r"the\s+([a-z]+)\s+block\s+is\s+on\s+the\s+table", fact)
            if m:
                x = m.group(1)
                goal["ontable"].add(x)
                blocks.add(x)

        if not blocks:
            blocks = self._extract_blocks_from_statement(self._extract_final_statement(scenario_context))

        return state, goal, sorted(self._normalize_blocks(blocks))

    def _blk_key(self, state):
        return tuple(sorted(state["on"])), tuple(sorted(state["ontable"])), state["holding"]

    def _blk_clear(self, state, blocks):
        blocked = {y for _, y in state["on"]}
        clear = set(blocks) - blocked
        if state["holding"] is not None:
            clear.discard(state["holding"])
        return clear

    def _blk_goal(self, state, goal):
        return goal["on"].issubset(state["on"]) and goal["ontable"].issubset(state["ontable"])

    def _blk_copy(self, state):
        return {"on": set(state["on"]), "ontable": set(state["ontable"]), "holding": state["holding"]}

    def _blk_goal_support_map(self, goal):
        support = {}
        for x, y in goal["on"]:
            support[x] = y
        return support

    def _blk_h(self, state, goal):
        unsat_on = len(goal["on"] - state["on"])
        unsat_ontable = len(goal["ontable"] - state["ontable"])
        wrong_on = sum(1 for pair in state["on"] if pair not in goal["on"])
        holding_penalty = 1 if state["holding"] else 0
        return 3 * unsat_on + unsat_ontable + wrong_on + holding_penalty

    def _blk_succ(self, state, goal, blocks):
        clear = self._blk_clear(state, blocks)
        holding = state["holding"]
        out = []
        goal_support = self._blk_goal_support_map(goal)

        if holding is not None:
            x = holding
            y = goal_support.get(x)
            if y is not None and y in clear and y != x:
                ns = self._blk_copy(state)
                ns["holding"] = None
                ns["on"].add((x, y))
                out.append((f"(mount_node {x} {y})", ns))
                return out

            ns = self._blk_copy(state)
            ns["holding"] = None
            ns["ontable"].add(x)
            out.append((f"(release_payload {x})", ns))
            return out

        unmount_candidates = []
        for x, y in sorted(state["on"]):
            if x in clear:
                if (x, y) in goal["on"]:
                    continue
                priority = (
                    0 if goal_support.get(x) not in (None, y) else 1,
                    x,
                    y,
                )
                unmount_candidates.append((priority, x, y))

        for _, x, y in sorted(unmount_candidates):
            ns = self._blk_copy(state)
            ns["on"].discard((x, y))
            ns["holding"] = x
            out.append((f"(unmount_node {x} {y})", ns))

        pickup_candidates = []
        for x in sorted(state["ontable"]):
            if x in clear:
                priority = (
                    0 if x in goal_support else 1,
                    x,
                )
                pickup_candidates.append((priority, x))

        for _, x in sorted(pickup_candidates):
            ns = self._blk_copy(state)
            ns["ontable"].discard(x)
            ns["holding"] = x
            out.append((f"(engage_payload {x})", ns))

        return out

    def _blk_succ_complete(self, state, _goal, blocks):
        clear = sorted(self._blk_clear(state, blocks))
        holding = state["holding"]
        out = []

        if holding is not None:
            x = holding

            for y in clear:
                if y != x:
                    ns = self._blk_copy(state)
                    ns["holding"] = None
                    ns["on"].add((x, y))
                    out.append((f"(mount_node {x} {y})", ns))

            ns = self._blk_copy(state)
            ns["holding"] = None
            ns["ontable"].add(x)
            out.append((f"(release_payload {x})", ns))
            return out

        for x, y in sorted(state["on"]):
            if x in clear:
                ns = self._blk_copy(state)
                ns["on"].discard((x, y))
                ns["holding"] = x
                out.append((f"(unmount_node {x} {y})", ns))

        for x in sorted(state["ontable"]):
            if x in clear:
                ns = self._blk_copy(state)
                ns["ontable"].discard(x)
                ns["holding"] = x
                out.append((f"(engage_payload {x})", ns))

        return out

    def _plan_blocks(self, scenario_context: str) -> list[str]:
        parsed = self._parse_blocks(scenario_context)
        if parsed is None:
            return []

        state0, goal, blocks = parsed
        if self._blk_goal(state0, goal):
            return []

        plan = self._best_first(
            state0,
            key_fn=self._blk_key,
            goal_fn=lambda s: self._blk_goal(s, goal),
            succ_fn=lambda s: self._blk_succ(s, goal, blocks),
            h_fn=lambda s: self._blk_h(s, goal),
            max_depth=max(22, 4 * max(1, len(goal["on"])) + 8),
            max_exp=150000,
        )
        if plan:
            return plan

        return self._best_first(
            state0,
            key_fn=self._blk_key,
            goal_fn=lambda s: self._blk_goal(s, goal),
            succ_fn=lambda s: self._blk_succ_complete(s, goal, blocks),
            h_fn=lambda s: self._blk_h(s, goal),
            max_depth=max(48, 8 * max(1, len(goal["on"])) + 16),
            max_exp=700000,
        )

    def solve(self, scenario_context: str, llm_engine_func) -> list:
        is_blocks = "set of blocks" in scenario_context.lower()

        if is_blocks:
            plan = self._plan_blocks(scenario_context)
            if plan:
                return plan

            blocks = self._extract_blocks_from_statement(self._extract_final_statement(scenario_context))
            prompt = (
                f"{scenario_context}\n\n"
                "Generate only the missing plan for the final [STATEMENT].\n"
                "Return one action per line and only these forms:\n"
                "(engage_payload x)\n"
                "(unmount_node x y)\n"
                "(release_payload x)\n"
                "(mount_node x y)\n"
                "Use only block names that appear in the statement.\n"
                "No explanations. No numbering. No extra words."
            )
            raw = llm_engine_func(prompt=prompt, system=self.system_prompt_blocks, max_new_tokens=240)
            extracted = self._extract_actions(raw, self._blk_patterns)
            return self._sanitize_block_actions(extracted, blocks)

        plan = self._plan_objects(scenario_context)
        if plan:
            return plan

        prompt = (
            f"{scenario_context}\n\n"
            "Generate only the missing plan for the final [STATEMENT].\n"
            "One action per line using only:\n"
            "(attack x)\n(succumb x)\n(feast x y)\n(overcome x y)\n"
            "No explanations or numbering."
        )
        raw = llm_engine_func(prompt=prompt, system=self.system_prompt_objects, max_new_tokens=220)
        return self._extract_actions(raw, self._obj_patterns)