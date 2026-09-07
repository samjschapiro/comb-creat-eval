"""Model display names, provider mapping and brand colours -- the single source of truth.

These four objects were duplicated across plot_radar, plot_profiles, make_composite_table and
make_appendix_tables. The copies drifted: every pool expansion left some of them behind, so tables
printed raw keys like "anthropic/claude-opus-4-8" while figures printed "claude-opus-4.8". They live
here now, in a module with NO matplotlib dependency, so the table scripts can import them too.
"""

LOGO_SLUG = {"openai": "openai", "anthropic": "claude", "google": "googlegemini", "x-ai": "xai",
             "deepseek": "deepseek", "qwen": "qwen", "z-ai": "zai", "meta": "meta"}
# microsoft and moonshotai have no logo asset on disk, so they get a brand colour and no mark.
# approximate brand colors; models sharing a provider get distinct shades of the same hue.
BRAND = {"openai": "#10A37F", "google": "#4285F4", "anthropic": "#D97757", "x-ai": "#1A1A1A",
         "qwen": "#615CED", "meta": "#0866FF", "deepseek": "#4D6BFE", "z-ai": "#2F7D6E",
         "microsoft": "#7FBA00", "moonshotai": "#6D28D9"}

DISPLAY = {
    "openai_gpt-5": "gpt-5", "openai_gpt-5-6-sol": "gpt-5.6-sol",
    "anthropic_claude-sonnet-4-5": "claude-sonnet-4.5", "google_gemini-3-1-pro-preview": "gemini-3.1-pro",
    "google_gemini-3-7-flash": "gemini-3.7-flash", "qwen_qwen3-max": "qwen3-max",
    "openai_gpt-5-2": "gpt-5.2", "openai_gpt-5-mini": "gpt-5-mini",
    "x-ai_grok-4-6": "grok-4.6", "x-ai_grok-4-5": "grok-4.5",
    "anthropic_claude-opus-5": "claude-opus-5", "anthropic_claude-sonnet-5": "claude-sonnet-5",
    "deepseek_deepseek-chat": "deepseek-chat", "deepseek_deepseek-r1": "deepseek-r1",
    "z-ai_glm-4-6": "glm-4.6", "google_gemini-3-flash-preview": "gemini-3-flash",
    "google_gemini-2-5-pro": "gemini-2.5-pro", "meta-llama_llama-3-3-70b-instruct": "llama-3.3-70b",
    "anthropic_claude-opus-4-6": "claude-opus-4.6", "anthropic_claude-opus-4-5": "claude-opus-4.5",
    "anthropic_claude-fable-5": "claude-fable-5",
    "google_gemini-2-5-flash": "gemini-2.5-flash", "meta-llama_llama-4-maverick": "llama-4-maverick",
    "qwen_qwen-2-5-72b-instruct": "qwen-2.5-72b", "deepseek_deepseek-chat-v3-0324": "deepseek-v3",
    "moonshotai_kimi-k2": "kimi-k2", "openai_gpt-4o-mini": "gpt-4o-mini", "openai_gpt-4-1": "gpt-4.1",
    "microsoft_phi-4": "phi-4", "z-ai_glm-4-5-air": "glm-4.5-air",
    # pool expansion 2026-09-06/07 (30 -> 35)
    "anthropic_claude-opus-4-8": "claude-opus-4.8", "anthropic_claude-opus-4-7": "claude-opus-4.7",
    "anthropic_claude-sonnet-4-6": "claude-sonnet-4.6", "anthropic_claude-fable-5-1": "claude-fable-5.1",
    "openai_gpt-6-astra-flex": "gpt-6-astra-flex",
}
def _provider(model_key):
    """The provider a model key belongs to. Falls back to the key's own prefix so a provider with no
    logo asset still gets its brand hue -- keying this on LOGO_SLUG silently greyed those models."""
    return next((p for p in LOGO_SLUG if model_key.startswith(p)),
                model_key.split("_", 1)[0] or None)

