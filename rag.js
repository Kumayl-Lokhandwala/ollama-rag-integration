import readline from 'readline';
import { RAGApplicationBuilder } from '@llm-tools/embedjs';
import { HNSWDb } from '@llm-tools/embedjs-hnswlib';
import { PdfLoader } from '@llm-tools/embedjs-loader-pdf';
import { Ollama } from '@llm-tools/embedjs-ollama';
import { OllamaEmbeddings } from '@llm-tools/embedjs-ollama';

const files = [
    './pdf1.pdf',
    './pdf2.pdf',
    './pdf3.pdf',
]

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

console.log("📚 Loading PDF-based RAG...");

const CONTEXT_GUARD_PROMPT = `
You are a document lookup system that only answers with exact quotes from the provided files.
Rules:
1. If the answer isn't a direct quote, respond: "Not found in documents"
2. Never add commentary, explanations or follow-ups
3. Keep responses under 20 words
4. Never reference the documents or your capabilities
`.trim();

const ragApplication = await new RAGApplicationBuilder()
  .setModel(new Ollama({
    modelName: "llama3.2",
    baseUrl: 'http://localhost:11434',
    system: CONTEXT_GUARD_PROMPT,
    temperature: 0.1,
    topK: 10,
    numCtx: 2048
  }))
  .setEmbeddingModel(new OllamaEmbeddings({
    model: 'nomic-embed-text',
    baseUrl: 'http://localhost:11434'
  }))
  .setVectorDatabase(new HNSWDb())
  .build();

// Load PDF documents
// ✅ FIX: Create PdfLoader
const loader = new PdfLoader();

for (const file of files) {
  const rawText = await loader.load(file);
  console.log(`🔍 Text extracted from ${file}:\n`, rawText.slice(0, 500));
  
  // ✅ Feed raw text into the vector DB
  await ragApplication.addDocuments([{
    content: rawText,
    metadata: { source: file }
  }]);
}



// Working Vector Database Inspection
async function inspectVectorDatabase() {
  console.log("\n🔍 Vector Database Inspection");
  try {
    // Basic info
    console.log(`ℹ️ Database Type: ${ragApplication.vectorDatabase.constructor.name}`);
    
    // Test search to verify functionality
    const testQuery = "university assistant";
    console.log(`🔎 Testing search for: "${testQuery}"`);
    
    const results = await ragApplication.query(testQuery);
    if (results?.content) {
      console.log("✅ Database is working. Sample result:");
      console.log(results.content.substring(0, 100) + "...");
    } else {
      console.log("⚠️ No results found - documents may not be properly indexed");
    }
    
  } catch (error) {
    console.error("⚠️ Inspection failed:", error.message);
  }
}

// Response validation
function isDocumentResponse(text) {
  const forbiddenPhrases = [
    "I", "you", "we", "us", "chatbot", "AI", "model",
    "assistant", "sorry", "apologize", "LLM", "as an AI"
  ];
  return text && !forbiddenPhrases.some(phrase => 
    text.toLowerCase().includes(phrase.toLowerCase())
  );
}

// Main chat interface
async function promptUser() {
  rl.question("🧠 You: ", async (input) => {
    const command = input.toLowerCase().trim();
    
    if (command === 'exit') {
      rl.close();
      return;
    } else if (command === 'debug') {
      await inspectVectorDatabase();
      promptUser();
      return;
    }

    try {
      const response = await ragApplication.query(input);
      let answer = "Not found in documents";
      
      if (response?.content && isDocumentResponse(response.content)) {
        answer = response.content
          .replace(/"/g, '')
          .replace(/\n/g, ' ')
          .trim()
          .split('.')[0];
      }
      
      console.log(`\n${answer}\n`);
    } catch (error) {
      console.log(`\n⚠️ Error: ${error.message}\n`);
    }

    promptUser();
  });
}

console.log("\n💬 Ready for queries. Commands:");
console.log("  - Ask a question about the documents");
console.log("  - Type 'debug' to verify database");
console.log("  - Type 'exit' to quit\n");

// Start chat interface
promptUser();