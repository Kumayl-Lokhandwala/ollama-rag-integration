import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { Ollama } from "@langchain/ollama";
import { retriever } from "./vector.js";
import { encode } from "gpt-tokenizer";
import readline from "readline/promises";
import { RunnableSequence } from "@langchain/core/runnables";

const model = new Ollama({
  baseUrl: process.env.OLLAMA_HOST,
  model: "llama3.2",
  temperature: 0.3, // More factual
});

// Prompt template
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

const prompt = ChatPromptTemplate.fromTemplate(template);

// Final runnable chain
const chain = RunnableSequence.from([
  prompt,
  model,
  new StringOutputParser()
]);

// CLI setup
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

async function main() {
  console.log("📚 Research Paper Analysis System");
  console.log("Initializing...");

  while (true) {
    const question = await rl.question("\nAsk about the research (q to quit): ");
    if (question.toLowerCase() === "q") break;

    const relevantDocs = await retriever.invoke(question);

    // Token truncation setup
    const MAX_TOKENS = 6000;
    let contextTexts = "";
    let totalTokens = 0;

    for (const doc of relevantDocs) {
      const chunk = `SOURCE: ${doc.metadata.source} (Chunk ${doc.metadata.chunkNumber}/${doc.metadata.totalChunks})\nCONTENT: ${doc.pageContent}\n\n---\n\n`;
      const chunkTokens = encode(chunk).length;

      if (totalTokens + chunkTokens > MAX_TOKENS) break;

      contextTexts += chunk;
      totalTokens += chunkTokens;
    }

    console.log("\n🔍 Searching papers...");
    const response = await chain.invoke({
      reviews: contextTexts,
      question: question,
    });

    console.log("\n🧠 RESPONSE:\n");
    console.log(response);

    console.log("\n📄 SOURCES USED:");
    relevantDocs.forEach(doc => {
      console.log(`- ${doc.metadata.source} (Chunk ${doc.metadata.chunkNumber})`);
    });
  }

  rl.close();
}

main().catch(console.error);
