from pathlib import Path
p = Path(r"frontend/src/shared/submissionExport.js")
text = p.read_text(encoding="utf-8")
text = text.replace('  const zipFiles = normalizeZipFiles(files);\n\n  zipFiles.forEach((file) => {', '  const zipFiles = [{ name: "submission/", content: "" }, ...normalizeZipFiles(files)];\n\n  zipFiles.forEach((file) => {')
p.write_text(text, encoding="utf-8")
