import Image from 'next/image'; // Импортируйте компонент Image
// import pic from '@/public/example.png'; // Импортируйте изображение
import { FaPlay } from "react-icons/fa";
import { FaEarthAsia } from "react-icons/fa6";

export default function LeftSide() {
    return (
        <div className="h-screen p-5 w-1/3 bg-gray-700 bg-opacity-20 backdrop-blur-md shadow-lg flex flex-col items-center justify-center">
            <div className=''>
                <button className=' text-xl bg-gray-500 bg-opacity-50 m-2 z-30 backdrop-blur-md rounded-xl p-3 absolute '>
                <FaPlay></FaPlay>
                </button>
                {/* <Image 
                    src={pic} 
                    alt="Example" 
                    layout="responsive" 
                    className="rounded-2xl" 
                /> */}
            </div>

            <div className='  flex flex-row  bg-black/20 w-full p-5 rounded-2xl m-5'>
                <div className='w-1/2 border-r-2 border-black border-dotted'>
                    <FaEarthAsia className='text-green-500 w-full h-full p-10'></FaEarthAsia>
                </div>
                
                <div className='w-1/2 p-2'>
                    <h1 className='text-2xl font-semibold mb-5'>In Earth conditions</h1>
                    <h1 className='text-2xl font-semibold'>Magnitude: 5.2</h1>
                    
                </div>
            </div>
        </div>
    );
}
