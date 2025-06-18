import React, { useState, useEffect, useRef } from "react";

function SingleSelectDropdown({ options, title, setSelected }) {
  const dropdownRef = useRef(null);
  const [isOpen, setIsOpen] = useState(false);
  const [selectedItem, setSelectedItem] = useState("");

  // useEffect(() => {
  //     setSelected(""); // reset on mount
  // }, [options]);

  useEffect(() => {
    if (setSelected) setSelected(selectedItem);
  }, [selectedItem]);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (!dropdownRef.current?.contains(event.target)) {
        setIsOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleSelect = (item) => {
    setSelectedItem(item);
    setIsOpen(false); // close dropdown after selection
  };

  return (
    <div ref={dropdownRef} className="mb-5">
      <div className="w-[18vw] relative z-20 m-3">
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="bg-[#6966FF] text-white py-2 px-4 rounded-full w-full text-left flex justify-between items-center"
          aria-expanded={isOpen}
        >
          {title}
          <span>{isOpen ? "▲" : "▼"}</span>
        </button>

        {isOpen && (
          <div className="absolute z-50 w-full bg-[#6966FF] border border-gray-300 mt-1 rounded-lg shadow-lg">
            <div className="max-h-48 overflow-y-auto">
              {options.map(({ label, value }, index) => (
                <div
                  key={index}
                  className="flex items-center p-2 hover:bg-gray-200 hover:text-black text-white"
                >
                  <input
                    type="checkbox"
                    id={`${title}-${index}`}
                    checked={selectedItem === value}
                    onChange={() => handleSelect(value)}
                    className="custom-checkbox mr-2"
                  />
                  <label htmlFor={`${title}-${index}`}>{label}</label>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default SingleSelectDropdown;
