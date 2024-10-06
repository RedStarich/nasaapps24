import Link from 'next/link';

const UniversityCard = ({ university }: { university: any }) => {
    return (
        <Link href={`/uni-list/${encodeURIComponent(university.name)}`}>
            <div className="bg-white shadow rounded-lg p-4 flex items-center border border-gray-300">
                <img src={university.logo_url} alt={university.name} className="w-32 h-32 object-cover rounded-md mr-4" />
                <div>
                    <h3 className="text-xl font-semibold text-gray-800">{university.name}</h3>
                    <p className="text-gray-600 mt-2">{university.description}</p>
                    <p className="text-blue-500 hover:underline mt-2 inline-block">Visit university page</p>
                </div>
            </div>
        </Link>
    );
};

export default UniversityCard;
