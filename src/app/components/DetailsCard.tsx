const DetailsCard = ({ university }: { university: any }) => {
    return (
      <div className="p-4 bg-gray-100 rounded-lg">
        <h2 className="text-xl font-semibold">Детали</h2>
        <ul className="mt-2 space-y-1">
          <li><strong>Год основания:</strong> {university.established_year}</li>
          <li><strong>Тип:</strong> {university.type}</li>
          <li><strong>Рейтинг:</strong> {university.ranking || 'н/д'}</li>
          <li><strong>Стоимость обучения:</strong> {university.tuition_fees ? `$${university.tuition_fees}` : 'н/д'}</li>
          <li><strong>Процент принятия:</strong> {university.acceptance_rate ? `${university.acceptance_rate}%` : 'н/д'}</li>
        </ul>
      </div>
    );
  };
  
  export default DetailsCard;
  