from pathlib import Path
p = Path(r"frontend/src/features/workspace/Workstation.jsx")
lines = p.read_text(encoding="utf-8").splitlines()
for idx, line in enumerate(lines):
    if 'className="ws-key"' in line and ' open' in line:
        lines[idx] = '        <span className="ws-key"><Keycap>Enter</Keycap> open</span>'
    if 'className="ws-key"' in line and ' keep' in line:
        lines[idx] = '        <span className="ws-key"><Keycap>Space</Keycap> keep</span>'
p.write_text('\n'.join(lines) + '\n', encoding="utf-8")
