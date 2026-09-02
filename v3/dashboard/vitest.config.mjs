import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    // The default forks pool times out starting workers on this Windows box;
    // threads starts clean. Nothing in the suite needs process isolation.
    pool: "threads",
    fileParallelism: false,
    // The whole-app render mounts two recharts figures in jsdom, which costs
    // several seconds of layout work before the first assertion runs. That is
    // the price of testing against the real committed artefacts rather than
    // fixtures, and it sits just over vitest's 5 s default.
    testTimeout: 20000,
  },
});
