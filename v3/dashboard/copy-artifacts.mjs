// Copies the repo's committed artifacts into public/ so dev and build serve
// the same files the panel can open in the repo. Run by predev/prebuild.
// Ported from ver1/dashboard unchanged (T-0127, #61) — only the source
// directory's contents changed shape (T-0126's contract), not this mechanism.
import { cpSync, mkdirSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const src = join(here, "..", "artifacts");
const dest = join(here, "public", "artifacts");

mkdirSync(dest, { recursive: true });
if (existsSync(src)) {
  cpSync(src, dest, { recursive: true });
  console.log(`artifacts copied: ${src} -> ${dest}`);
} else {
  console.warn("no artifacts/ directory found — dashboard will show empty states");
}
