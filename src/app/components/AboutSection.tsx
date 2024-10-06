import React from 'react';
import Image from 'next/image';

const AboutSection = () => {
  return (
    <section id="about" className="relative py-20 bg-cover bg-center" style={{ backgroundImage: 'url(/images/background-about.png)' }}>
      <div className="container mx-auto px-4 flex flex-col lg:flex-row items-center">
        <div className="lg:w-1/2 text-white z-10">
          <h2 className="text-3xl font-bold mb-4">О Unimori.ai</h2>
          <p className="mb-4">
            Unimori.ai – это инновационная платформа, созданная для поддержки студентов. Мы предоставляем полную информацию о вашем университете, включая расписания, мероприятия, ресурсы и многое другое. Наша цель – сделать вашу университетскую жизнь проще и продуктивнее.
          </p>
        </div>
        <div className="lg:w-1/2 mt-8 lg:mt-0 flex justify-center z-10">
          <Image src="/images/dog.png" alt="Unimori Dog" width={300} height={300} />
        </div>
      </div>
      <div className="absolute inset-0 bg-black opacity-50 z-0"></div>
    </section>
  );
};

export default AboutSection;
