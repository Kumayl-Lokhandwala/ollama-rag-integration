import fs from "fs/promises";
import path from "path";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';

export async function setupVectorStore() {
  const pdfDir = "./papers";
  const files = await fs.readdir(pdfDir);

  const allDocs = [];

  for (const file of files) {
    if (path.extname(file) === ".pdf") {
      const loader = new PDFLoader(path.join(pdfDir, file), {
        splitPages: false, // true = each page is a separate doc
      });
      const docs = await loader.load();
      allDocs.push(...docs);
    }
  }

  const embeddings = new OllamaEmbeddings({
    model: "mxbai-embed-large",
    baseUrl: process.env.OLLAMA_HOST,
  });

  const vectorStore = await FaissStore.fromDocuments(allDocs, embeddings);

  await vectorStore.save("./faiss_store");

  return vectorStore.asRetriever({ k: 5 });
}

export const retriever = await setupVectorStore();
