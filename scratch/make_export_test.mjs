import fs from 'node:fs/promises';
import { buildSubmissionCsv, makeSubmissionZip } from '../frontend/src/shared/submissionExport.js';

const items = Array.from({ length: 100 }, (_, idx) => ({
  videoKey: 'L21_V024',
  folderKey: 'L21_a',
  frameKey: String(19504 + idx),
  globalFrameId: 19504 + idx,
  frameName: `L21_V024_${String(19504 + idx).padStart(6, '0')}`,
  backend: { frame_idx: 19504 + idx, video_id: 'L21_V024' },
}));
const csv = buildSubmissionCsv(items, 'kis');
const blob = makeSubmissionZip([{ name: 'query-p1-1-kis.csv', content: csv, queryType: 'kis' }]);
await fs.writeFile('scratch/aic_export_test.zip', Buffer.from(await blob.arrayBuffer()));
console.log(csv.split('\n').length);
console.log(csv.split('\n').slice(0, 3).join('\n'));
