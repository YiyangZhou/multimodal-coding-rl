"""Render the README's native SVG flow diagrams (no dependencies)."""
from pathlib import Path
from html import escape
OUT = Path(__file__).resolve().parents[1] / 'docs'
INK, MUTED, LINE = '#1b2730', '#687882', '#a3b2bb'
TEAL, PURPLE = '#087f91', '#7160bc'

class Diagram:
    def __init__(self, name, title, subtitle, height=560):
        self.name, self.h = name, height
        self.s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="{height}" viewBox="0 0 1440 {height}" role="img" aria-labelledby="title desc">', f'<title id="title">{escape(title)}</title><desc id="desc">{escape(subtitle)}</desc>', '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M1 1 L9 5 L1 9" fill="none" stroke="#a3b2bb" stroke-width="1.5"/></marker></defs>', f'<rect width="1440" height="{height}" rx="22" fill="#f7f8fa"/>', '<g font-family="Arial,Helvetica,sans-serif">']
        self.text(40, 45, title, 25, INK, 600)
        self.text(40, 76, subtitle, 15)
    def text(self, x, y, value, size=14, color=MUTED, weight=400, anchor='start'):
        self.s.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{escape(value)}</text>')
    def path(self, d, dashed=False, arrow=True, color=LINE):
        self.s.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.7" stroke-linejoin="round"'+(' stroke-dasharray="5 5"' if dashed else '')+(' marker-end="url(#arrow)"' if arrow else '')+'/>')
    def box(self, x,y,w,label,title,detail='',tone=None,h=104):
        color = TEAL if tone=='teal' else PURPLE if tone=='purple' else MUTED
        fill = '#eaf6f7' if tone=='teal' else '#f0edfb' if tone=='purple' else '#ffffff'
        self.s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}" stroke="#dce3e9"/>')
        self.text(x+18,y+27,label.upper(),11,color,600)
        self.text(x+18,y+56,title,18,INK,500)
        if detail:self.text(x+18,y+81,detail,13)
    def note(self,t):self.text(40,self.h-22,t,13)
    def save(self):
        self.s.append('</g></svg>'); OUT.mkdir(exist_ok=True); (OUT/(self.name+'-flow.svg')).write_text('\n'.join(self.s)+'\n')

d=Diagram('architecture','One policy. Multiple ways to solve.','Multimodal tasks → direct or agentic rollouts → verified rewards → NeMo RL updates.',600)
# Edges are laid out first, keeping the policy feedback separate from artifact flow.
d.path('M240 252 H280 V162 H320');d.path('M280 252 V324 H320')
d.path('M550 162 H590 V252 H650');d.path('M550 324 H590 V252',arrow=False)
d.path('M880 252 H930');d.path('M1150 252 H1200')
d.path('M435 376 V406');d.path('M320 450 H295 V350 H320')
d.path('M1300 304 V532 H268 V124 H320',dashed=True)
d.path('M268 324 H320',dashed=True)
d.box(40,200,200,'Input adapters','Task + references','Images · text · code')
d.box(320,110,230,'Single-turn','Direct generation','One model answer','teal')
d.box(320,272,230,'Multi-turn','Agent harness','OpenCode · Codex · Hermes','purple')
d.box(320,406,230,'Remote sandbox','Tools + observations','Execute · render · inspect')
d.box(650,200,230,'Rollout contract','Trajectory capture','Tokens · logprobs · lineage','teal')
d.box(930,200,220,'Task verifier','Execute + score','Final artifact → reward')
d.box(1200,200,200,'NeMo RL','Grouped update','Advantages → policy','purple')
d.text(760,342,'Complete prompt groups',14,TEAL,500)
d.text(760,366,'Validity masks + length rules',13)
d.text(735,521,'Versioned policy sync → generation workers',13,PURPLE)
d.note('Private judge references stay outside model prompts. The Gym service and sandbox runtime are external dependencies.')
d.save()

d=Diagram('single-turn','01 / Single-turn','A direct answer becomes an executable artifact; verification supplies the training reward.',500)
for a,b in [(240,290),(500,550),(760,810),(1030,1080)]:d.path(f'M{a} 234 H{b}')
d.path('M1240 286 V402 H395 V286',dashed=True)
d.box(40,182,200,'Input','Task prompt','Allowed references')
d.box(290,182,210,'Policy','Generate answer','Exact tokens + logprobs','teal')
d.box(550,182,210,'Final artifact','Extract code','Preserve original output')
d.box(810,182,220,'Verification','Execute + render','Task-specific scoring')
d.box(1080,182,320,'NeMo RL','Reward → group → update','Model failures ≠ service failures','purple')
d.text(870,157,'Sandbox / verifier',13)
d.text(645,390,'Policy sync for subsequent rollouts',13,PURPLE)
d.note('No harness tool loop. Truncation and infrastructure validity are checked before a sample contributes to training.')
d.save()

d=Diagram('multi-turn','02 / Multi-turn','The harness runs a bounded model–tool loop; the final artifact is scored independently.',600)
d.path('M230 252 H280');d.path('M500 252 H550');d.path('M770 252 H820');d.path('M1050 252 H1100')
d.path('M660 200 V160 H390 V200')
d.path('M390 304 V426 H550')
d.path('M1100 280 H1080 V478 H850 V426 H790',dashed=True)
d.path('M1280 304 V538 H255 V252 H280',dashed=True)
d.box(40,200,190,'Input','Task + assets','Prompt / workspace')
d.box(280,200,220,'Shared policy','Model call','Captured sampled tokens','teal')
d.box(550,200,220,'Harness + sandbox','Execute tools','OpenCode / Codex / Hermes','purple')
d.box(820,200,230,'Terminal output','Final artifact','Completion / budget stop')
d.box(1100,200,300,'Verification + NeMo RL','Score → grouped update','Independent execution + reward')
d.box(550,374,240,'Capture gateway','Training trajectory','Logprobs · request lineage','teal')
d.text(472,149,'Tool observations → next model call',13,PURPLE)
d.text(820,466,'Reward + eligibility',13)
d.text(810,527,'Updated policy → next rollout',13,PURPLE)
d.text(410,367,'Record every model call',13)
d.note('Only eligible model tokens are learning targets. Tool-output tokens remain context, not policy targets.')
d.save()

d=Diagram('mixed','03 / Mixed rollout','Direct and agentic paths converge on one reward contract and one trainable policy.',690)
# Four parallel lanes make same-prompt sibling allocation explicit.
ys=[126,242,358,474]
d.path('M230 352 H270');d.path('M490 352 H530 V178 H580')
for y in ys[1:]:d.path(f'M530 {y+52} H580')
d.path('M530 178 V526',arrow=False)
for y in ys:d.path(f'M830 {y+52} H880',arrow=False)
d.path('M880 178 V526',arrow=False);d.path('M880 352 H940');d.path('M1160 352 H1210')
d.path('M1305 404 V625 H555 V178 H580',dashed=True)
for y in ys[1:]:d.path(f'M555 {y+52} H580',dashed=True)
d.box(40,300,190,'Input','Prompt group','Stable prompt identity')
d.box(270,300,220,'Routing strategy','Assign before sampling','Stable through retries')
for y,label,title,detail,tone in [(126,'Direct','Single-turn','2 siblings / prompt','teal'),(242,'Agentic','OpenCode','2 siblings / prompt','purple'),(358,'Agentic','Codex','2 siblings / prompt','purple'),(474,'Agentic','Hermes','2 siblings / prompt','purple')]:d.box(580,y,250,label,title,detail,tone)
d.box(940,300,220,'Shared contract','Verify + reward','Final artifacts + masks')
d.box(1210,300,190,'NeMo RL','Group + update','Shared policy','purple')
d.text(40,148,'GROUP ROUTING',11,MUTED,600)
d.text(40,175,'One route for all siblings',15,INK)
d.text(40,208,'SIBLING MIX',11,PURPLE,600)
d.text(40,235,'Same prompt, four routes',15,INK)
d.text(40,260,'8 siblings: 2 per route',14)
d.text(945,614,'Versioned policy sync to every route',13,PURPLE)
d.note('The 2 + 2 + 2 + 2 allocation describes sibling-mixed mode. Group routing chooses one supported route per prompt group.')
d.save()
