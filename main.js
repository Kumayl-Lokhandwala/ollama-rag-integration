import { Ollama } from "@langchain/community/llms/ollama";
import { PromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { retriever } from "./vector.js";
import readline from "readline/promises";

const model = new Ollama({
  baseUrl: process.env.OLLAMA_HOST,
  model: "llama3.2",
  temperature: 0.3 // Lower temperature for more factual responses
});

const template = `
You are a research assistant analyzing academic papers. 
Provide detailed, accurate answers based on the provided context.

Relevant paper excerpts:
{reviews}

Question: {question}

Guidelines:
- Only follow the information in research paper
- If information is not found in the context, say the information is not available
- Do not make assumptions or provide personal opinions
- Use the context to answer the question directly
- Be precise and cite sources when possible
- If information is not found, say so explicitly
- Maintain an academic tone
- Highlight key findings when relevant
`;

const prompt = PromptTemplate.fromTemplate(template);
const chain = prompt.pipe(model).pipe(new StringOutputParser());

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

async function main() {
  console.log("Research Paper Analysis System");
  console.log("Initializing... (This may take a few minutes for first run)");
  
  while (true) {
    const question = await rl.question("\nAsk about the research (q to quit): ");
    if (question.toLowerCase() === "q") break;

    const relevantDocs = await retriever.invoke(question);
    const contextTexts = relevantDocs.map(doc => 
      `SOURCE: ${doc.metadata.source} (Chunk ${doc.metadata.chunkNumber}/${doc.metadata.totalChunks})\n` +
      `CONTENT: ${doc.pageContent}`
    ).join("\n\n---\n\n");
    
    console.log("\nSearching papers...");
    const response = await chain.invoke({
      reviews: contextTexts,
      question: question
    });

    console.log("\nRESPONSE:");
    console.log(response);
    
    console.log("\nSOURCES USED:");
    relevantDocs.forEach(doc => {
      console.log(`- ${doc.metadata.source} (Chunk ${doc.metadata.chunkNumber})`);
    });
  }

  rl.close();
}

main().catch(console.error);