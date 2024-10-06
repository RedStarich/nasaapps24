// 'use client'
// import { Bar } from 'react-chartjs-2';
// import {
//     Chart as ChartJS,
//     CategoryScale,
//     LinearScale,
//     BarElement,
//     Title,
//     Tooltip,
//     Legend
// } from 'chart.js';

// ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

// export default function VerticalBarChart() {
//     const data = {
//         labels: ['2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023'],
//         datasets: [
//             {
//                 label: 'Earthquakes per Year',
//                 data: [500, 600, 800, 1500, 2000, 2500, 3000, 2300, 2800, 3200],
//                 backgroundColor: '#FF5733', 
//             },
//         ],
//     };

//     const options = {
//         responsive: true,
//         maintainAspectRatio: false,
//         plugins: {
//             legend: {
//                 display: false,
//             },
//             title: {
//                 display: true,
//                 text: 'Earthquake Histogram (Last Decade)',
//             },
//         },
//         scales: {
//             x: {
//                 grid: {
//                     display: false, // Убираем вертикальные линии
//                 },
//             },
//             y: {
//                 beginAtZero: true,
//                 ticks: {
//                     stepSize: 500,
//                 },
//                 grid: {
//                     display: true, // Включаем горизонтальные линии
//                     color: 'rgba(0, 0, 0, 0.1)', // Цвет пунктирных линий
//                     borderDash: [4, 2], // Делаем линии пунктирными
//                 },
//             },
//         },
//     };

//     return (
//         <div className="w-full h-full">
//             <Bar data={data} options={options} />
//         </div>
//     );
// }
