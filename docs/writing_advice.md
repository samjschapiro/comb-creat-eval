# Writing Advice for ML Papers

Condensed from Neel Nanda, "Highly Opinionated Advice on How to Write ML Papers" (May 2025). Source: https://www.alignmentforum.org/posts/eJGptPbbFPZGLpjsp/highly-opinionated-advice-on-how-to-write-ml-papers

---

## The Essence of a Paper

A paper presents a **narrative of 1-3 specific concrete claims** that build to a useful takeaway. Everything else exists to support this. The two pillars:

1. **Communicate the key idea**: motivate it, contextualize in literature, explain precisely
2. **Provide rigorous evidence** that the claims are true

The north star: ensure the reader **understands**, **remembers**, and **believes** the narrative.

## Crafting a Narrative

- Compress your research into 1-3 claims. Readers rarely take away more than a few sentences. **Choose those sentences carefully.**
- Claims should fit a **cohesive theme** — papers are far easier to understand if there's a coherent narrative, not a grab-bag of unconnected ideas.
- Adjust claim confidence to match evidence strength: existence-proof, systematic, hedged, narrow, or guarantee.
- Questions to find your narrative: What would be most exciting to show someone? Why should anyone care? What was hard or non-trivial?

## Rigorous Evidence

- **Quality over quantity**: one compelling, hard-to-deny experiment beats many mediocre ones.
- **Red-team aggressively**: Assume you've made a mistake — where? Assume there's a hole — find it. Get others to weigh in.
- **Good experiments distinguish between hypotheses** — results should vary significantly depending on which hypothesis is true.
- **Can you trust your results?** Ask: "How surprised would I be if this turned out to be bullshit due to a bug, noise, or misunderstanding?"
- **Statistical rigor**: Don't use p < .05 as threshold. Papers reporting central findings at .01 < p < .05 usually fail to replicate. Be skeptical of anything that isn't p < .001 for exploratory work.
- **Ablation studies**: Remove one component at a time. Show which parts are necessary.
- **Baselines are crucial**: Show your method beats plausible alternatives, not just that it "works at all." Put meaningful effort into making baselines strong.
- **Diverse lines of evidence**: Several qualitatively different experiments pointing to the same conclusion are more robust than many similar ones.
- **Avoid cherry-picking**: Note how cherry-picked qualitative evidence is. Provide randomly selected examples for context.
- **Track pre/post-hoc analysis**: Pre-specified experiments confirmed by results are more compelling than post-hoc interpretation.
- **Reproducibility**: Share code. Ensure it runs on a fresh machine. Write a README. Create a demo notebook.

## The Writing Process: Compress Then Iteratively Expand

### Time allocation
Spend comparable time on each of: **the abstract, the intro, the figures, and everything else** — they have roughly the same number_of_readers * time_to_read.

### Compress first
- Verbally describe your work to someone. Ask what's most interesting.
- Plan out a talk.
- Force yourself to identify: (1) the 1-3 claims, (2) why they matter, (3) the crucial evidence for each.

### Then iteratively expand
1. **Compressed bullet point narrative** — are you happy this captures the narrative?
2. **Bullet point outline of intro** — claims, why they matter, high-level evidence
3. **Bullet point outline of full paper** — experiments, results, methodology, limitations. Every part should have a clear role; if you can't answer "what goes wrong if I cut this?", cut it.
4. **Results and figures** — collect results, make draft figures. Does this support your narrative?
5. **First draft** — flesh into prose with full technical detail
6. **Edit** — repeatedly pass over, polish, tighten, get feedback

Get feedback at each stage. Feedback on a bullet outline is faster and more useful than on 8 pages of prose.

## Paper Structure

### Abstract
- **Sentence 1**: Uncontroversially true statement situating the subfield
- **Sentence 2**: The need/gap/problem your paper addresses (conveys motivation)
- **Sentence 3**: Your key contribution and why it's exciting (lose nuance, this is OK)
- **Sentence 4** (optional): Clarifying details on the claim
- **Next sentences**: Key evidence or additional claims, one sentence per idea
- **Include a concrete metric or result** to show results are real and substantial
- **Final 1-2 sentences**: Why the paper matters, implications, broader context
- Abstracts should be **as accessible as possible** — minimize jargon
- The reader is coming in cold — help them orient fast

### Introduction
- **Paragraph 1 (Context)**: Topic, motivating question, why it matters. Cite liberally to establish the field is real and the problem matters.
- **Paragraph 2 (Technical background)**: What do we know? What techniques does this rest on? Situate in the broader strategic picture.
- **Paragraph 3 (Key contribution)**: What exactly is the main claim? Add nuance, detail, context.
- **Paragraph 3.5 (Our case)**: Summarize the most critical evidence.
- **Paragraph 4 (Impact)**: Implications, takeaways, who should take different actions.
- **Contributions list**: End with bullet points of concise claim descriptions with key evidence.
- The introduction is a **self-contained paper summary** — don't worry about "spoiling" or repetition. With complex ideas, repeat in varied ways so it sticks.

### Figures
- **Incredibly important**. Many readers only skim figures and captions.
- Ask: "What exactly should someone take away from this?" Choose visualization that emphasizes the takeaway.
- Ask: "Why does this tie back to my core claims? Which parts do I want to draw attention to?"
- **Annotate** important lines or points. Emphasize key lines (dark vs. light/low opacity).
- Include axis titles, clear captions, readable tick labels, good legends.
- For heatmaps: white-at-zero with dark-at-max for positive data; diverging (RdBu) for positive/negative with meaningful zero.
- Avoid red/green for key information (4% of people are red-green colorblind).
- Combine several key graphs into one figure for Figure 1.
- Explanatory diagrams (not just data plots) can be very effective as Figure 1.

### Main Body (Background, Methods, Results)
- Communicate experiments and results in **precise technical detail** so researchers can understand exactly what you did and draw their own conclusions.
- Multiple layers of abstraction: key background context, what results are and their significance, what you actually did, why it was reasonable, specifics of technical choices.
- Structure: Background (define terminology!), Methods, Results.
- If experiments for each claim are different, give each its own section rather than methods→results. People forget the first method before seeing its results.

### Discussion / Limitations
- **Acknowledge limitations honestly** — you know more about your work's weaknesses than readers do. Papers that don't do this are substantially weaker.
- Don't try to sound maximally exciting. Competent researchers see through this, and respect honest acknowledgment of limitations.
- Discuss broader implications, future work, reflections.
- Conclusions are often useless if the intro already explained things well. You can skip it.

### Related Work
- Important to explain why your work is different from similar prior work.
- Prefer as penultimate section, not second section (unless literature heavily motivates the paper).
- Contextualizing is important, but the intro should cover the key citations.

### Appendices
- Everything that doesn't fit in the main paper. Lower standard, rarely read.
- Think of it as a much longer paper where the main body is the highlights reel.
- Don't stress polishing appendices too much.

## Novelty

- **Be extremely clear about what is and is not novel**, especially in intro and related work.
- Cite the most relevant papers and explain precisely why your work is different.
- The same paper can seem arrogant or modestly incremental depending on how novelty claims are presented.
- Contextualizing within existing literature is **particularly crucial for experienced researchers** — helps them quickly see what's interesting rather than blurring into superficially similar papers.

## Common Pitfalls

- **Unnecessary complexity and verbosity**: If a reader doesn't understand your paper, they ignore it or assume it's BS. Best papers take simple techniques and apply them carefully. Be precise but accessible. Minimize jargon. You get points for quality insights, not sounding fancy.
- **Not prioritizing writing**: Writing is a major multiplier on impact. Switch to writing mode a month before deadline. Iterate extensively.
- **Illusion of transparency**: You have months of context; the reader has none. Address misconceptions and misunderstandings even though they feel obvious to you.
- **Inform, not persuade**: Avoid overclaiming or ignoring limitations. Scientific integrity gains respect from the researchers who matter.

## References

- Delimit all AI-generated references in the .bib with a comment before and after that indicates the reference was AI-generated and needs to be verified by a human. E.g.: `% --- AI-GENERATED REFERENCE (VERIFY) ---` before and after each entry.

## Tacit Knowledge

Consider including as appendix A or blog post:
- What was hard and what steps got it to work
- Ways experiments caught fire and how you fixed them
- Fuzzy big-picture intuitions you can't fully defend but believe after months of work
- Common misconceptions about your results
- Advice for replication, especially hyperparameters and fiddly bits
- Exciting future work directions
