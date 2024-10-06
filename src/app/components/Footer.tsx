import React from 'react';

const Footer = () => {
  return (
    <footer className="bg-blue-600 text-white py-8">
      <div className="container mx-auto text-center">
        <nav className="mb-4">
          <a href="#" className="ml-4">Privacy</a>
          <a href="#" className="ml-4">FAQ</a>
          <a href="#" className="ml-4">Shipping and Payment</a>
          <a href="#" className="ml-4">Partners</a>
          <a href="#" className="ml-4">Blog</a>
          <a href="#" className="ml-4">Contacts</a>
        </nav>
        <p>&copy; 2024 Unimori.ai. All rights reserved.</p>
      </div>
    </footer>
  );
};

export default Footer;
