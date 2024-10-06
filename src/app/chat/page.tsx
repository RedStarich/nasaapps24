'use client';

import { useState } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import { Loader2, Send } from 'lucide-react';

const ChatGPT: React.FC = () => {
  const [question, setQuestion] = useState<string>('');
  const [answer, setAnswer] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleQuestionChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setQuestion(e.target.value);
  };

  const handleFormSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post('/api/gemini/route', { question });
      setAnswer(response.data.answer);
    } catch (err) {
      setError('An error occurred while processing your request.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen w-full relative bg-[url('https://hebbkx1anhila5yf.public.blob.vercel-storage.com/image-XNUvuviJWor2hbYAJD0el04KaVwUNQ.png')] bg-cover bg-center">
      <div className="container mx-auto p-4 max-w-4xl relative z-10">
        <div className="bg-black/50 backdrop-blur-md p-6 rounded-lg border border-gray-700 shadow-xl">
          <h2 className="text-2xl font-bold text-white mb-4">Chat with GPT</h2>
          <form onSubmit={handleFormSubmit} className="space-y-4">
            <input
              type="text"
              placeholder="Ask a question..."
              value={question}
              onChange={handleQuestionChange}
              className="w-full h-12 bg-gray-800 text-white border border-gray-700 rounded px-3 py-2"
            />
            <button
              type="submit"
              disabled={loading}
              className="flex justify-center gap-2 items-center mx-auto shadow-xl text-lg bg-gray-50 backdrop-blur-md lg:font-semibold isolation-auto border-gray-50 before:absolute before:w-full before:transition-all before:duration-700 before:hover:w-full before:-left-full before:hover:left-0 before:rounded-full before:bg-emerald-500 hover:text-gray-50 before:-z-10 before:aspect-square before:hover:scale-150 before:hover:duration-700 relative z-10 px-4 py-2 overflow-hidden border-2 rounded-full group"
            >
              {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Send className="mr-2 h-4 w-4" />}
              {loading ? 'Processing...' : 'Ask'}
            </button>
          </form>

          {loading && <div className="text-white mt-4">Loading...</div>}
          {error && <div className="text-red-500 mt-4">{error}</div>}

          {answer && (
            <div className="mt-8">
              <h3 className="text-xl font-semibold mb-4 text-white">Answer:</h3>
              <div className="space-y-4 bg-white p-4 rounded-lg border border-gray-700">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[rehypeRaw]}
                  components={{
                    a: ({ node, ...props }) => <a {...props} className="text-black underline" />,
                    code: ({ node, ...props }) => <code {...props} className="bg-gray-700 p-1 rounded text-black" />,
                    pre: ({ node, ...props }) => <pre {...props} className="bg-gray-700 p-2 rounded overflow-x-auto text-black" />,
                    table: ({ node, ...props }) => <table {...props} className="border-collapse w-full text-white" />,
                    th: ({ node, ...props }) => <th {...props} className="border border-gray-600 px-4 py-2 bg-gray-700 font-bold text-center" />,
                    td: ({ node, ...props }) => <td {...props} className="border border-gray-600 px-4 py-2" />,
                  }}
                >
                  {answer}
                </ReactMarkdown>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ChatGPT;
