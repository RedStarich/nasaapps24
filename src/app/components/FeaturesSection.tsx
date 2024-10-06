import React from 'react';

const features = [
  { title: 'Полная база данных университетов', description: 'Получите доступ к информации о всех университетах в вашем регионе.' },
  { title: 'Расписание занятий', description: 'Легко находите и следите за своим расписанием.' },
  { title: 'Мероприятия и события', description: 'Будьте в курсе всех университетских мероприятий и событий.' },
  { title: 'Ресурсы и материалы', description: 'Найдите учебные материалы и ресурсы, необходимые для успешной учебы.' },
  { title: 'Поддержка сообщества', description: 'Общайтесь и сотрудничайте с другими студентами через нашу платформу.' },
];

const FeaturesSection = () => {
  return (
    <section id="features" className="relative py-20" style={{ backgroundImage: 'url(/images/background.png)', backgroundSize: 'cover', backgroundPosition: 'center' }}>
      <div className="container mx-auto text-center relative z-10">
        <h2 className="text-3xl font-bold mb-8 text-white">Что мы предлагаем?</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <div key={index} className="p-4 bg-white bg-opacity-90 rounded shadow">
              <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
              <p>{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
      <div className="absolute inset-0 bg-black opacity-30 z-0"></div>
    </section>
  );
};

export default FeaturesSection;
