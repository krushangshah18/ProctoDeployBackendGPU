import { fileURLToPath } from "url";
import path from "path";

// Explicitly set the base directory so @tailwindcss/postcss resolves
// the tailwindcss package from frontend/node_modules rather than
// defaulting to process.cwd() which can point to the repo root.
const __dirname = path.dirname(fileURLToPath(import.meta.url));

const config = {
  plugins: {
    "@tailwindcss/postcss": {
      base: __dirname,
    },
  },
};

export default config;
