const ContactCard = ({ university } : { university: any }) => {
    return (
      <div className="p-4 bg-gray-100 rounded-lg">
        <h2 className="text-xl font-semibold">Contact Information</h2>
        <ul className="mt-2 space-y-1">
          <li><strong>Email:</strong> <a href={`mailto:${university.contact_email}`} className="text-blue-500 hover:underline">{university.contact_email}</a></li>
          <li><strong>Phone:</strong> {university.contact_phone || 'н/д'}</li>
        </ul>
      </div>
    );
  };
  
  export default ContactCard;
  