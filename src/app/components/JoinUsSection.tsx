import React from 'react';
import Image from 'next/image';
import Link from 'next/link';

const JoinUsSection = () => {
  return (
    <section id="join-us" className="py-20 bg-white relative overflow-hidden">
      <div className="absolute inset-0 flex justify-center items-center">
        <div className="w-[95%]">
          <Image src="/images/join-us.png" alt="Background" layout="responsive" width={916} height={326} />
        </div>
      </div>
      <div className="container mx-auto text-center relative z-10">
        <h2 className="text-3xl font-bold mb-4">Присоединяйтесь к Unimori.ai сегодня</h2>
        <p className="text-lg mb-8">Создайте бесплатную учетную запись и начните пользоваться всеми преимуществами нашей платформы.</p>
        <button className="bg-blue-600 text-white px-6 py-3 rounded-lg text-lg font-semibold"><Link href='/uni-list'>Залетай</Link></button>
      </div>
    </section>
  );
};

export default JoinUsSection;
