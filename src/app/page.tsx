'use client';
import { useState } from 'react'
import Link from 'next/link'
import { Search, BookOpen, MessageSquare, AlertTriangle, Brain, ChevronRight } from 'lucide-react'

export default function LandingPage() {
  const [email, setEmail] = useState('')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    // Handle newsletter signup
    console.log('Signed up with:', email)
    setEmail('')
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 text-white">
      <header className="p-4 bg-slate-900">
        <nav className="container mx-auto flex items-center justify-between">
          <Link href="/" className="text-xl font-bold md:text-2xl">AI Research Hub</Link>
          <ul className="flex space-x-6">
            {['Papers', 'Topics', 'Chat', 'About', 'Contact'].map((item) => (
              <li key={item}>
                <Link href={item.toLowerCase() === 'chat' ? '/chat' : `#${item.toLowerCase()}`} className="hover:text-blue-400 transition-colors">
                  {item}
                </Link>
              </li>
            ))}
          </ul>
        </nav>
      </header>
      
      <main className="container mx-auto px-4 py-12">
        <section className="text-center mb-16">
          <h1 className="text-5xl font-bold mb-4">Explore AI Risks and Intelligence</h1>
          <p className="text-xl mb-8">Access cutting-edge research papers and interact with our AI-powered research assistant</p>
          <Link href="/chat" className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-full inline-flex items-center transition-colors">
            Try Our AI Chat <ChevronRight className="ml-2" />
          </Link>
        </section>

        <section className="grid md:grid-cols-2 lg:grid-cols-3 gap-8 mb-16">
          <div className="bg-slate-800 p-6 rounded-lg">
            <Search className="w-12 h-12 mb-4 text-blue-400" />
            <h2 className="text-2xl font-bold mb-2">Search Papers</h2>
            <p>Find relevant research papers on AI risks and intelligence from top institutions.</p>
          </div>
          <div className="bg-slate-800 p-6 rounded-lg">
            <BookOpen className="w-12 h-12 mb-4 text-blue-400" />
            <h2 className="text-2xl font-bold mb-2">Read Summaries</h2>
            <p>Get concise summaries of complex research papers to quickly grasp key concepts.</p>
          </div>
          <div className="bg-slate-800 p-6 rounded-lg">
            <MessageSquare className="w-12 h-12 mb-4 text-blue-400" />
            <h2 className="text-2xl font-bold mb-2">Chat with AI</h2>
            <p>Interact with our AI-powered assistant to get insights from the research papers.</p>
          </div>
        </section>

        <section className="bg-slate-800 p-8 rounded-lg mb-16">
          <h2 className="text-3xl font-bold mb-4">Featured Topics</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            {['AI Safety', 'Machine Ethics', 'Superintelligence', 'AI Governance', 'Existential Risk', 'AI Alignment'].map((topic) => (
              <Link key={topic} href={`#${topic.toLowerCase().replace(' ', '-')}`} className="bg-slate-700 p-4 rounded-lg hover:bg-slate-600 transition-colors">
                {topic}
              </Link>
            ))}
          </div>
        </section>

        <section className="text-center mb-16">
          <h2 className="text-3xl font-bold mb-4">Why AI Research Hub?</h2>
          <div className="grid md:grid-cols-2 gap-8">
            <div className="flex flex-col items-center">
              <AlertTriangle className="w-16 h-16 mb-4 text-yellow-400" />
              <h3 className="text-2xl font-bold mb-2">Understand AI Risks</h3>
              <p>Stay informed about potential risks associated with advanced AI systems and their implications for society.</p>
            </div>
            <div className="flex flex-col items-center">
              <Brain className="w-16 h-16 mb-4 text-green-400" />
              <h3 className="text-2xl font-bold mb-2">Explore AI Intelligence</h3>
              <p>Dive deep into the latest research on artificial intelligence, machine learning, and cognitive architectures.</p>
            </div>
          </div>
        </section>

        <section className="bg-slate-800 p-8 rounded-lg">
          <h2 className="text-3xl font-bold mb-4 text-center">Stay Updated</h2>
          <p className="text-center mb-4">Subscribe to our newsletter for the latest research papers and AI insights.</p>
          <form onSubmit={handleSubmit} className="flex max-w-md mx-auto">
            <input
              type="email"
              placeholder="Enter your email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="flex-grow px-4 py-2 rounded-l-lg text-slate-900"
              required
            />
            <button type="submit" className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-r-lg transition-colors">
              Subscribe
            </button>
          </form>
        </section>
      </main>

      <footer className="bg-slate-900 text-center p-4 mt-12">
        <p>&copy; 2023 AI Research Hub. All rights reserved.</p>
      </footer>
    </div>
  )
}