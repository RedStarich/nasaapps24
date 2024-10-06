'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { HiMenu, HiX } from 'react-icons/hi';
import Image from 'next/image';

const Header = () => {
  const [menuOpen, setMenuOpen] = useState(false);

  const toggleMenu = () => {
    setMenuOpen(!menuOpen);
  };

  const handleScroll = (event: React.MouseEvent<HTMLAnchorElement, MouseEvent>, targetId: string) => {
    event.preventDefault();
    const targetElement = document.getElementById(targetId);
    if (targetElement) {
      targetElement.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <header className="bg-white shadow-md sticky top-0 z-50">
      <div className="container mx-auto flex justify-between items-center p-4">
        <Link href="/" passHref>
          <p className="flex items-center">
            <Image src="/images/main-logo.png" alt="Unimori.ai" width={150} height={42} />
          </p>
        </Link>
        <div className="hidden md:flex space-x-8 items-center">
          <a href="/uni-list">Список университетов</a>
          <a href="/uni-list">Отбор </a>
       </div>
        <div className="md:hidden">
          <button onClick={toggleMenu} className="text-blue-600 focus:outline-none">
            {menuOpen ? <HiX size={28} /> : <HiMenu size={28} />}
          </button>
        </div>
      </div>
      {menuOpen && (
        <nav className="md:hidden bg-white shadow-lg">
          <ul className="space-y-4 p-4">
            <li>
              <a href="#about" onClick={(e) => { handleScroll(e, 'about'); setMenuOpen(false); }} className="text-gray-700 block">О нас</a>
            </li>
            <li>
              <a href="#features" onClick={(e) => { handleScroll(e, 'features'); setMenuOpen(false); }} className="text-gray-700 block">Особенности</a>
            </li>
            <li>
              <a href="#how-it-works" onClick={(e) => { handleScroll(e, 'how-it-works'); setMenuOpen(false); }} className="text-gray-700 block">Как это работает</a>
            </li>
            <li>
              <a href="#join-us" onClick={(e) => { handleScroll(e, 'join-us'); setMenuOpen(false); }} className="text-gray-700 block">Присоединяйтесь</a>
            </li>
          </ul>
        </nav>
      )}
    </header>
  );
};

export default Header;
