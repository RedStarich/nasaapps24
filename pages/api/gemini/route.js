import { GoogleGenerativeAI } from "@google/generative-ai";
import dotenv from 'dotenv';
import data from '../../../data/data3.json'; // Adjust the path as necessary

dotenv.config();

const apiKey = process.env.NEXT_PUBLIC_GEMINI_API_KEY || "";
if (!apiKey) {
  throw new Error('API_KEY is not defined in the environment variables');
}

const genAI = new GoogleGenerativeAI(apiKey);

const generationConfig = {
  temperature: 1,
  topP: 0.95,
  topK: 64,
  maxOutputTokens: 20000,
  responseMimeType: "text/plain",
};

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method Not Allowed' });
  }

  const { question } = req.body;

  if (!question) {
    return res.status(400).json({ error: 'Question is required' });
  }

  // Use default data if specific context is not provided
  const context = `Here is some information:\n${JSON.stringify(data, null, 2)}\n\n`;

  try {
    const model = genAI.getGenerativeModel({
      model: "gemini-1.5-flash",
      systemInstruction: `You are a professor in Artificial Intelligence. Answer questions based on the provided data. Stay professional and polite. Responses should be respectful, constructive, and focused on improvement suggestions. Your responses should be supported with valid and authorized references to the relevant academic papers. If you cannot answer to a question, say so shortly and clearly. Use markdown for responses, ensuring proper formatting for headings, lists, links, and code snippets. Style links with TailwindCSS classes "text-blue-500 underline".`,
    });

    const chatSession = model.startChat({
      generationConfig,
      history: [
        {
          role: "user",
          parts: [{ text: `${context} ${question}` }],
        },
      ],
    });

    const result = await chatSession.sendMessage(question);
    console.log("API Response:", result); // Debugging line

    res.status(200).json({ answer: result.response.text() });
  } catch (error) {
    console.error("Error generating response:", error); // Debugging line
    res.status(500).json({ error: 'An error occurred while processing the request' });
  }
}
