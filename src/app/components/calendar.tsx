'use client'

import React, { useState } from 'react';

const Calendar = () => {
  const [selectedDate, setSelectedDate] = useState(null);
  const daysInMonth = 30; // You can change this to the number of days in the desired month
  const numbers = Array.from({ length: daysInMonth }, (_, index) => index + 1); // Replace with your specific numbers if needed

  const handleDateClick = (date:any) => {
    setSelectedDate(date);
  };

  return (
    <div className="calendar grid grid-cols-7 gap-4 p-4">
      {numbers.map((number, index) => (
        <div
          key={index}
          className={`date-item p-4 text-center border rounded cursor-pointer ${
            selectedDate === number ? 'bg-blue-500 text-white' : 'bg-gray-200'
          }`}
          onClick={() => handleDateClick(number)}
        >
          {number}
        </div>
      ))}
    </div>
  );
};

export default Calendar;
