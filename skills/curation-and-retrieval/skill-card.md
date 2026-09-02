## Description: <br>
GPU-accelerated video and image curation via NVIDIA Cosmos Curator inside Physical AI Data Factory, turning raw video and image collections into curated, training-ready datasets with KPI-driven, distribution-aware, and restrictive curation support. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to configure and run NVIDIA Cosmos Curator pipelines (split, filter, caption, embed, dedup, shard, image annotate) and PAIDF Data Mining nearest-neighbor retrieval to produce training-ready datasets for physical AI. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [API key, Cloud Credentials] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Running Pipelines](references/running-pipelines.md) <br>
- [Cosmos Curator](references/cosmos-curator.md) <br>
- [FFmpeg Sidecar](references/ffmpeg-sidecar.md) <br>
- [Calibration Config](references/calibration-config.md) <br>
- [Configuration Decision Tree](references/configuration-decision-tree.md) <br>
- [Capabilities](references/capabilities.md) <br>
- [Video Curation](references/video-curation.md) <br>
- [Image Curation](references/image-curation.md) <br>
- [Data Mining](references/data-mining.md) <br>
- [Curation Retrieval Workflow](references/curation-retrieval-workflow.md) <br>
- [Distribution Analysis](references/distribution-analysis.md) <br>
- [Distribution-Aware Curation](references/distribution-aware-curation.md) <br>
- [Restrictive Curation](references/restrictive-curation.md) <br>
- [SAM3 Config](references/sam3-config.md) <br>
- [Gotchas](references/gotchas.md) <br>
- [KPI Metrics](references/kpi-metrics.md) <br>
- [Context Understanding](references/context-understanding.md) <br>
- [Video Lake Curation](references/video-lake-curation.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
13 evaluation tasks (13 positive) from skill-evaluator-dataset-snapshot/1, each in an isolated sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether the skill is safe to use: no unsafe operations, secret leakage, or unauthorized access. <br>
- Correctness: Checks whether the answer is correct against the reference answer. <br>
- Discoverability: Checks whether the right skill was found and executed when needed. <br>
- Effectiveness: Checks whether the skill helped complete the user's goal and expected workflow (goal_accuracy 50% + behavior_check 50%). <br>
- Efficiency: Checks whether the skill avoided wasted tool or skill usage via routing quality and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Verifies absence of unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Verifies final-answer correctness against the reference answer. <br>
- `skill_execution`: Verifies the expected skill was found and executed. <br>
- `goal_accuracy`: Verifies whether the user's goal was achieved. <br>
- `behavior_check`: Verifies whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Verifies routing quality, workspace-aware skill reads, and productive tool use. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 51% → 90% (+39 points) | 48% → 84% (+36 points) |
| Security | 92% → 100% (+8 points) | 81% → 100% (+19 points) |
| Correctness | 46% → 100% (+54 points) | 57% → 95% (+38 points) |
| Discoverability | 38% → 84% (+45 points) | 30% → 65% (+35 points) |
| Effectiveness | 49% → 88% (+39 points) | 41% → 84% (+43 points) |
| Efficiency | 29% → 78% (+49 points) | 29% → 76% (+47 points) |

## Skill Version(s): <br>
1.1.0 (source: frontmatter, pyproject.toml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
