# Inference Research

An Astro static site for documenting research into Windows-native LLM inference: direct Safetensors access, C# and Direct3D compute shaders, layer streaming, and GPU memory management.

## Local development

```sh
npm install
npm run dev
```

Create a production build with `npm run build`; preview it locally with `npm run preview`.

## Writing research posts

Add a Markdown file under `src/content/research/` with this frontmatter:

```yaml
---
title: "A research note"
description: "A short summary for the research index."
order: 6
track: "Systems"
status: "Planned"
---
```

Use ordinary Markdown for prose and code, `$...$` / `$$...$$` for KaTeX math, and fenced `mermaid` blocks for diagrams. Posts marked `Planned` are presented as outlines rather than completed work.

## Azure Static Web Apps

The site is statically generated into `dist/`. The GitHub Actions workflow in `.github/workflows/azure-static-web-apps.yml` builds the Astro site and deploys it to Azure Static Web Apps on pushes to `main`; pull requests get preview environments.

1. Create an Azure Static Web App and connect it to this GitHub repository, or create it separately and choose the GitHub Actions deployment option.
2. Add the deployment token from the Static Web App to the repository's GitHub Actions secrets as `AZURE_STATIC_WEB_APPS_API_TOKEN`.
3. Push to `main` to publish the site.

The workflow uses `/` as the app location, no API location, and `dist` as the build output. No Azure resources are provisioned by this repository.
