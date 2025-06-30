import fs from "fs/promises";
import path from "path";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { OllamaEmbeddings } from "@langchain/ollama";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

export async function setupVectorStore() {
  const pdfDir = "./papers";
  const files = await fs.readdir(pdfDir);

  const allDocs = [];

  for (const file of files) {
    if (path.extname(file) === ".pdf") {
      const loader = new PDFLoader(path.join(pdfDir, file), {
        splitPages: false, // load whole PDF
      });
      const docs = await loader.load();

      // 🔥 Split text to avoid embedding size issues
      const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200,
      });

      const splitDocs = await splitter.splitDocuments(docs);
      allDocs.push(...splitDocs);
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
//     const reviews = relevantDocs.map(doc => doc.pageContent).join("\n\n");