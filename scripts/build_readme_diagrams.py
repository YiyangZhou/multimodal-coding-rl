"""Build the README's editable, dependency-free SVG architecture diagrams."""
from pathlib import Path
from html import escape

OUT = Path(__file__).resolve().parents[1] / 'docs'
COLORS = {'teal': '#087f8c', 'purple': '#7357c7', 'blue': '#3268bb', 'amber': '#ac7215'}

def diagram(name, title, subtitle, rows, footer):
    height = 180 + len(rows) * 185
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="{height}" viewBox="0 0 1440 {height}" role="img" aria-labelledby="title desc">',
           f'<title id="title">{escape(title)}</title><desc id="desc">{escape(subtitle + ". " + footer)}</desc>',
           '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8" fill="none" stroke="#8996ab" stroke-width="1.3"/></marker></defs>',
           f'<rect width="1440" height="{height}" rx="22" fill="#f6f8fc"/>',
           '<style>text{font-family:Arial,Helvetica,sans-serif} .title{font-size:30px;font-weight:700;fill:#18243a}.sub{font-size:17px;fill:#53627b}.heading{font-size:20px;font-weight:700;fill:#18243a}.body{font-size:17px;fill:#44536a}.label{font-size:15px;font-weight:700;letter-spacing:1px}</style>',
           f'<text x="36" y="48" class="title">{escape(title)}</text>',
           f'<text x="36" y="80" class="sub">{escape(subtitle)}</text>']
    for r, (label, color, cards) in enumerate(rows):
        y = 116 + r * 185
        c = COLORS[color]
        svg.append(f'<text x="36" y="{y}" class="label" fill="{c}">{escape(label.upper())}</text>')
        width = (1368 - 32 * (len(cards)-1)) / len(cards)
        for i, (heading, lines) in enumerate(cards):
            x = 36 + i * (width + 32)
            svg += [f'<rect x="{x}" y="{y+16}" width="{width}" height="132" rx="12" fill="white" stroke="#dce3ee"/>',
                    f'<rect x="{x}" y="{y+28}" width="4" height="28" rx="2" fill="{c}"/>',
                    f'<text x="{x+18}" y="{y+47}" class="heading">{escape(heading)}</text>']
            for j, line in enumerate(lines):
                svg.append(f'<text x="{x+18}" y="{y+76+j*23}" class="body">{escape(line)}</text>')
            if i < len(cards)-1:
                svg.append(f'<path d="M{x+width+5} {y+82} H{x+width+25}" stroke="#8996ab" stroke-width="2" marker-end="url(#arrow)"/>')
    svg += [f'<text x="36" y="{height-25}" class="sub">{escape(footer)}</text>', '</svg>']
    OUT.mkdir(exist_ok=True)
    (OUT / f'{name}.svg').write_text('\n'.join(svg) + '\n')

diagram('architecture', 'Multimodal Coding RL', 'Task adapters, rollout routes, verification and policy updates share one training pipeline.', [
('01 / Generate', 'teal', [('Canonical task', ['Prompt + allowed input assets', 'Stable task and prompt IDs', 'Private judge references held back']), ('Route the prompt', ['Single-turn or agentic', 'OpenCode / Codex / Hermes', 'Group or sibling-level assignment']), ('Capture the rollout', ['Model tokens and behavior logprobs', 'Tool calls and terminal artifacts', 'Request lineage and stop reason'])]),
('02 / Verify and learn', 'purple', [('Independent verification', ['Execute in the configured sandbox', 'Render images, animations or 3D', 'Apply the task-specific verifier']), ('Reward and eligibility', ['Raw reward + configured penalties', 'Separate model and service failures', 'Apply truncation / validity masks']), ('NeMo RL update', ['Assemble complete prompt groups', 'Compute advantages and policy loss', 'Checkpoint, metrics and recovery'])])], 'Configured generation capacity and optimizer batch size are independent. Public runtime compatibility is documented separately.')

diagram('single-turn', 'Single-turn rollout', 'One direct model answer; execution and scoring occur after generation.', [
('01 / Answer', 'teal', [('Prepare input', ['Task prompt + permitted references', 'Tokenizer and generation settings', 'Stable prompt-group identity']), ('Generate once', ['No agent harness or tool loop', 'Capture exact sampled tokens', 'Record behavior logprobs']), ('Extract final artifact', ['Task-specific answer extraction', 'Retain original answer and code', 'Record completion / truncation'])]),
('02 / Evaluate', 'blue', [('Execute and render', ['Fresh verification context', 'Compile or run the submitted code', 'Collect artifacts and diagnostics']), ('Score the result', ['Task-specific quality metrics', 'Model execution failure: zero', 'Infrastructure failure: excluded']), ('Train eligible groups', ['Apply configured masks / penalties', 'Retain complete sibling groups', 'Update the shared policy'])])], 'Judge-only reference assets are not model inputs. A failed execution and an invalid infrastructure result are different outcomes.')

diagram('multi-turn', 'Agentic rollout', 'The selected harness controls the tool loop; the model remains the trainable policy.', [
('01 / Interaction loop', 'purple', [('Choose a harness', ['OpenCode, Codex or Hermes', 'Task prompt + sandbox workspace', 'Configured call and token budgets']), ('Model ↔ tools', ['Model proposes code or actions', 'Harness executes permitted tools', 'Observations inform the next call']), ('Finalize the artifact', ['Stop on completion or budget', 'Keep code and interaction history', 'Record the terminal status'])]),
('02 / Training contract', 'teal', [('Capture each model call', ['Exact tokens + behavior logprobs', 'Preserve request / response lineage', 'Separate tool observations']), ('Verify final output', ['Independent task execution', 'Render and score final artifacts', 'Classify infrastructure failures']), ('Build policy samples', ['Train on eligible model tokens', 'Tool-output tokens are not targets', 'Group rewards feed advantages'])])], 'Tool feedback belongs to the rollout context. Final verification supplies the reward; it does not rewrite the model answer.')

diagram('mixed', 'Mixed rollout', 'Two routing strategies share task adapters, reward contracts and a single trainable policy.', [
('A / Prompt-group routing', 'blue', [('Select a prompt group', ['Sample from the configured task mix', 'Choose one permitted route', 'Keep the route stable on retry']), ('Generate siblings', ['All siblings use the selected route', 'Direct or one supported harness', 'Preserve common prompt identity']), ('Complete the group', ['Verify each terminal artifact', 'Apply validity and length rules', 'Submit an eligible complete group'])]),
('B / Same-prompt sibling routing', 'purple', [('One prompt, eight siblings', ['2 × single-turn', '2 × OpenCode', '2 × Codex + 2 × Hermes']), ('Execute all four routes', ['Stable sibling-to-route assignment', 'Each route uses its own tool flow', 'Same underlying trainable policy']), ('One grouped update stream', ['Common reward contract', 'Compare eligible sibling outcomes', 'Log task and route breakdowns'])])], 'Route support is task-specific. Ready-first scheduling selects complete groups; it does not select answers by reward.')
