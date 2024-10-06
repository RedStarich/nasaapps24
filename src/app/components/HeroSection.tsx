import React from 'react';
import Link from 'next/link';

const HeroSection = () => {
  return (
    <section className="bg-white py-20">
      <div className="container mx-auto text-center">
        <h1 className="text-4xl font-bold text-blue-600 mb-4">
          Unimori.Ai – платформа для поддержки студентов
        </h1>
        <p className="text-lg text-gray-700 mb-8">
          Найдите всю необходимую информацию о вашем университете в одном месте
        </p>
        <button className="bg-blue-600 text-white px-6 py-3 rounded-lg text-lg font-semibold">
          <Link href='uni-list'>Найти информацию</Link>
        </button>
      </div>
    </section>
  );
};

export default HeroSection;
