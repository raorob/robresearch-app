import { defineCollection, z } from "astro:content";
import { glob } from "astro/loaders";

const posts = defineCollection({
	loader: glob({ base: "./src/content/research", pattern: "**/*.md" }),
	schema: z.object({
		title: z.string(),
		description: z.string(),
		order: z.number(),
		track: z.string(),
		status: z.enum(["In progress", "Planned"]),
	}),
});

export const collections = { posts };
