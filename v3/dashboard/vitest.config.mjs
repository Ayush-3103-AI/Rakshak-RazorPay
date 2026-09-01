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
  },
});
