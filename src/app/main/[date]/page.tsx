import Calendar from "../../components/calendar";
import LeftSide from "../../components/leftSide";
// import VerticalBarChart from "../../components/pic";
import Catalog from "../../components/catalog";


export default function Main({params}:any){

    const arr = [
        {
          title: 'Date',
          data: '8/02/23'
        },
        {
          title: 'Time',
          data: '2:30 PM'
        },
        {
          title: 'Magnitude',
          data: '5.4'
        },
        {
          title: 'Duration',
          data: '35 s'
        },
        
        {
          title: 'Depth',
          data: '36 (km)'
        },
        
      ]


    return(
        <div className="text-2xl bg-gradient-to-b flex flex-row from-blue-500 via-blue-300 to-blue-200 backdrop-blur-sm">
            <div className="bg-opacity-20 w-2/3 flex flex-col items-center justify-around h-screen bg-white backdrop-blur-md shadow-lg">
            <div className="h-[60vh] w-full p-4 rounded-3xl overflow-hidden flex relative"> 
                <h1 className="absolute z-10 text-lg m-5 bg-white/30 font-semibold backdrop-blur-md p-4 rounded-xl">
                enough to destroy New York
                </h1>
                
                {/* Используем absolute для контейнера с данными */}
                <div className="absolute bottom-10 left-10 right-10 border-white/30 border-2 flex flex-row items-center justify-around z-10 text-lg m-5 bg-white/20 font-semibold backdrop-blur-md p-4 rounded-xl">
                {arr.map((item: any, index: number) => (
                    <div key={index} className="flex p-5 flex-col">
                    <span className="text-lg font-medium text-gray-300">{item.title}</span>
                    <h4 className="text-2xl">{item.data}</h4>
                    </div>
                ))}
                </div>

                <img
                className={`h-full w-full rounded-xl shake`} // эффект тряски на изображении
                src="https://news.mit.edu/sites/default/files/images/202409/MIT-MissMars-01-press.jpg"
                alt="mars-lol"
                />
            </div>


                <div className="h-1/3 p-4 w-full">
                {/* <VerticalBarChart /> */}
                </div>
                
            </div>
            <LeftSide />
            
            </div>
    )
}