"""Cost-accounting checks — these numbers are what the customer sees, so pin them."""

from dexa_platform.gateway import pricing


def test_qwen_tokens_far_below_openai_for_a_screenshot():
    w, h = 1280, 800
    q = pricing.image_tokens("dexa-cua-vlm", w, h)
    o = pricing.image_tokens("gpt-4o", w, h)
    om = pricing.image_tokens("gpt-4o-mini", w, h)
    assert 200 < q < 1200          # Qwen merges patches -> few visual tokens
    assert o == 85 + 170 * 6       # 1280x800 -> 6 tiles on OpenAI's grid
    assert om > 30000              # 4o-mini's image multiplier is enormous
    assert q < o                    # and Qwen bills fewer image tokens than 4o too


def test_qwen_respects_pixel_budget():
    # a 4K screenshot must clamp to the max_pixels budget, not blow up linearly
    small = pricing.qwen_vision_tokens(1280, 800)
    big = pricing.qwen_vision_tokens(3840, 2160)
    assert big <= 324 + 1          # (1024/28-ish grid)/merge, budget-capped
    assert big >= small - 1 or big <= 324


def test_compare_reports_real_savings_vs_4o():
    s = pricing.compare([(1280, 800)], text_in_tokens=120, out_tokens=20, baseline_model="gpt-4o")
    assert s.saved_usd > 0
    assert s.x_cheaper > 1
    d = s.as_dict()
    assert d["baseline_model"] == "gpt-4o"
    assert 0 < d["dexa_cost_usd"] < d["baseline_cost_usd"]


def test_compare_vs_mini_is_dramatic():
    # documents/screenshots on 4o-mini are pathologically expensive; savings should be huge
    s = pricing.compare([(1280, 800)], text_in_tokens=120, out_tokens=20, baseline_model="gpt-4o-mini")
    assert s.x_cheaper > 5


def test_cost_is_monotonic_in_output_tokens():
    a = pricing.request_cost("dexa-cua-vlm", [(1280, 800)], 100, 10)
    b = pricing.request_cost("dexa-cua-vlm", [(1280, 800)], 100, 500)
    assert b.usd > a.usd
