"""Detector captions independent of the simulator's training object catalog."""

from __future__ import annotations

import re


def goal_caption(instruction: str, caption: str = "") -> str:
    """Accept arbitrary English target descriptions, plus common Chinese targets.

    A supplied English caption supports Chinese descriptions without pretending
    that a fixed dictionary is an open-vocabulary translator. Keep attributes
    and spatial descriptions. This function never resolves a simulator ID.
    """
    text = (caption or instruction).strip().lower()
    if not text:
        raise ValueError("目标描述不能为空。")
    if re.search(r"[\u4e00-\u9fff]", text):
        text = re.sub(r"^(?:请|帮我|请帮我)?(?:抓取|抓住|抓|拿起|拿|捡起)", "", text)
        text = re.sub(r"(?:放入|放进|放到).*$", "", text).strip(" 。！")
        aliases = {
            "橘子": "orange fruit", "橙子": "orange fruit",
            "小黄鸭": "yellow rubber duck", "黄鸭": "yellow rubber duck",
            "蓝色方块": "blue cube", "蓝方块": "blue cube",
            "红色圆柱": "red cylinder", "红圆柱": "red cylinder",
            "苹果": "apple", "香蕉": "banana", "杯子": "cup", "瓶子": "bottle",
        }
        if text not in aliases:
            raise ValueError("此中文描述需要 --caption 提供英文目标描述（保留颜色、形状或方位）。")
        text = aliases[text]
    else:
        text = re.sub(r"^(?:please\s+)?(?:pick\s+up|pick|grab|grasp|take|lift)\s+", "", text)
        text = re.sub(r"^(?:the|an|a)(?:\s+|$)", "", text)
        text = re.split(r"\s+(?:and\s+then|then)\s+", text)[0]
        text = re.split(r"\s+(?:and\s+)?(?:put|place|drop)\s+", text)[0]
        text = text.strip(" .!。！")
        if text == "orange":
            text = "orange fruit"
    if not text or text in {"it", "object", "something"}:
        raise ValueError("请明确描述一个目标物体。")
    if re.search(r"\b(?:and|or)\b|[,;\n]", text):
        raise ValueError("一次抓取需要一个目标描述。")
    return text.rstrip(".") + "."
