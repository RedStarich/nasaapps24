'use client';

import { useEffect, useState, useMemo } from 'react';
import UniversityCard from './UniCard'; // Import the new component

interface University {
  _id: string;
  name: string;
  location: string;
  description: string;
  logo_url: string;
  website_url: string;
  established_year: number;
  type: string;
  ranking: number;
  tuition_fees: number;
  acceptance_rate: number;
  contact_email: string;
  contact_phone: string;
  alternate_names?: string[];
}

const Loader: React.FC = () => (
  <div className="flex justify-center items-center h-screen">
    <img src="https://media1.tenor.com/m/OWI_ai132b0AAAAC/wink-dog.gif" alt="Loading..." />
  </div>
);

const UniversitiesList: React.FC = () => {
  const [universities, setUniversities] = useState<University[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    const fetchUniversities = async () => {
      try {
        const res = await fetch('/api/universities/route');
        if (!res.ok) {
          throw new Error('Failed to fetch universities');
        }
        const data = await res.json();
        if (data && Array.isArray(data.universities)) {
          setUniversities(data.universities);
        } else {
          throw new Error('Invalid data format');
        }
      } catch (err) {
        setError((err as Error).message);
      } finally {
        setLoading(false);
      }
    };

    fetchUniversities();
  }, []);

  const filteredUniversities = useMemo(() =>
    universities.filter((uni) =>
      uni.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (uni.alternate_names && uni.alternate_names.some(name => name.toLowerCase().includes(searchQuery.toLowerCase())))
    ), [universities, searchQuery]);

  if (loading) {
    return <Loader />;
  }

  if (error) {
    return <div className="text-red-500">Error: {error}</div>;
  }

  if (!universities.length) {
    return <div>No universities found.</div>;
  }

  return (
    <div className="container mx-auto px-4">
      <div className="flex flex-col justify-center items-center text-center my-8">
        <img src="/images/siba-uni-list.png" alt="Unimori.ai Logo" className="w-20 h-20 md:w-32 md:h-32" />
        <h1 className="text-3xl md:text-4xl font-bold mb-2">Привет, я Сиба Юни 🧡💙</h1>
        <p className="text-lg md:text-xl">Я могу помочь тебе с подбором университета!</p>
      </div>
      <div className="flex justify-center mb-8">
        <input
          type="text"
          placeholder="Начни поиск"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="p-2 md:p-4 border border-gray-300 rounded w-full max-w-lg"
        />
      </div>
      <ul className="space-y-4">
        {filteredUniversities.map((uni) => (
          <li key={uni._id}>
            <UniversityCard university={uni} />
          </li>
        ))}
      </ul>
    </div>
  );
};

export default UniversitiesList;
