# React + TypeScript + Vite

This template provides a minimal setup to get React working in Vite with HMR and some Oxlint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the Oxlint configuration

If you are developing a production application, we recommend enabling type-aware lint rules by installing `oxlint-tsgolint` and editing `.oxlintrc.json`:

```json
{
  "$schema": "./node_modules/oxlint/configuration_schema.json",
  "plugins": ["react", "typescript", "oxc"],
  "options": {
    "typeAware": true
  },
  "rules": {
    "react/rules-of-hooks": "error",
    "react/only-export-components": ["warn", { "allowConstantExport": true }]
  }
}
```

See the [Oxlint rules documentation](https://oxc.rs/docs/guide/usage/linter/rules) for the full list of rules and categories.

---

# Iba Lens mobile

On-device malaria smear triage: React + TypeScript, ONNX Runtime Web (WASM), Capacitor iOS shell.
Cell detection (`src/lib/segment.ts`, a port of `segment()` in `src/smearscan.py`) and the 9 MB MobileNetV2
(`public/model/`, exported by `src/export_onnx.py`) both run on the device, so it works offline.

```bash
npm install
npm run dev                 # http://localhost:5173  (add ?parity to compare against the Python pipeline)
npm run build && npx cap sync ios
npx cap open ios            # or: xcodebuild ... -scheme App, then launch in the Simulator
```

Parity (`src/parity.ts` vs `public/demo/reference.json` from `src/dump_reference.py`): on all 8 demo slides the
cell counts, flagged counts and slide bands match Python; max per-cell probability difference is 0.0007.
The per-cell heatmap is occlusion sensitivity (Grad-CAM needs gradients, which the exported graph doesn't carry).
