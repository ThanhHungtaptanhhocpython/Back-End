from pathlib import Path
p = Path(r"frontend/src/features/selection/SelectionTray.jsx")
text = p.read_text(encoding="utf-8")
text = text.replace('import { CloseOutlined, ExportOutlined } from "@ant-design/icons";', 'import { CloseOutlined, ExportOutlined, ThunderboltOutlined } from "@ant-design/icons";')
text = text.replace('export default function SelectionTray({ keptItems, onRemove, onClear, onExport, onOpen, trayRef }) {', 'export default function SelectionTray({ keptItems, onRemove, onClear, onExport, onOpenBatch, onOpen, trayRef }) {')
text = text.replace('''          <button className="ws-btn small export" onClick={onExport}>
            <ExportOutlined /> Export Submission
          </button>''', '''          <button className="ws-btn small" style={{ color: "#2563eb", borderColor: "#8fb2ff" }} onClick={onOpenBatch} title="Paste all queries and export one ZIP">
            <ThunderboltOutlined /> Batch Submit
          </button>
          <button className="ws-btn small export" onClick={onExport}>
            <ExportOutlined /> Export Submission
          </button>''')
p.write_text(text, encoding="utf-8")

w = Path(r"frontend/src/features/workspace/Workstation.jsx")
text = w.read_text(encoding="utf-8")
text = text.replace('import ExportModal from "./ExportModal";\nimport ShortcutOverlay', 'import ExportModal from "./ExportModal";\nimport BatchQueryModal from "./BatchQueryModal";\nimport ShortcutOverlay')
text = text.replace('  const [exportDefaultSource, setExportDefaultSource] = useState("results");', '  const [exportDefaultSource, setExportDefaultSource] = useState("results");\n  const [batchOpen, setBatchOpen] = useState(false);')
text = text.replace('''        onExport={openExport}
        onOpen={openReview}''', '''        onExport={openExport}
        onOpenBatch={() => setBatchOpen(true)}
        onOpen={openReview}''')
text = text.replace('''        searchType={activeTab?.searchType || "TEXT"}
        defaultSource={exportDefaultSource}''', '''        tabs={tabs}
        searchType={activeTab?.searchType || "TEXT"}
        defaultSource={exportDefaultSource}''')
text = text.replace('''      />
      <ShortcutOverlay open={showShortcuts} powerUser={powerUser} onTogglePowerUser={togglePowerUser} onClose={toggleHelp} />''', '''      />
      <BatchQueryModal open={batchOpen} onClose={() => setBatchOpen(false)} toast={toast} />
      <ShortcutOverlay open={showShortcuts} powerUser={powerUser} onTogglePowerUser={togglePowerUser} onClose={toggleHelp} />''')
text = text.replace('<Keycap>â†µ</Keycap>', '<Keycap>Enter</Keycap>')
text = text.replace('<Keycap>â£</Keycap>', '<Keycap>Space</Keycap>')
w.write_text(text, encoding="utf-8")
