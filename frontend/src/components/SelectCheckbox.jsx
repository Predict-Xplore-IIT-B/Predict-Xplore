import React, { useState, useEffect, useRef } from 'react';

function SelectCheckbox({ options, title }) {
    const dropdownRef = useRef(null);
    const [classesOpen, setClassesOpen] = useState(false);
    const [selectedClasses, setSelectedClasses] = useState(options);

    useEffect(() => {
        setSelectedClasses(options);
    }, [options]);

    useEffect(() => {
        const handleClickOutside = (event) => {
            if (!dropdownRef.current?.contains(event.target)) {
                setClassesOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    const toggleClass = (className) => {
        setSelectedClasses((prev) =>
            prev.includes(className)
                ? prev.filter((c) => c !== className)
                : [...prev, className]
        );
    };

    const toggleAllClasses = () => {
        if (selectedClasses.length === options.length) {
            setSelectedClasses([]);
        } else {
            setSelectedClasses(options);
        }
    };

    return (
        <div ref={dropdownRef} className="mb-5">
            <div className="w-[18vw] relative z-20 m-3">
                <button
                    onClick={() => setClassesOpen(!classesOpen)}
                    className="bg-[#6966FF] text-white py-2 px-4 rounded-full w-full text-left flex justify-between items-center"
                    aria-expanded={classesOpen}>
                    {title}
                    <span>{classesOpen ? "▲" : "▼"}</span>
                </button>

                {classesOpen && (
                    <div className="absolute z-50 w-full bg-[#6966FF] border border-gray-300 mt-1 rounded-lg shadow-lg">
                        <div className="max-h-48 overflow-y-auto">
                            {options.map((cls, index) => (
                                <div
                                    key={index}
                                    className="flex items-center p-2 hover:bg-gray-200 hover:text-black text-white"
                                >
                                    <input
                                        type="checkbox"
                                        id={`class-${index}`}
                                        checked={selectedClasses.includes(cls)}
                                        onChange={() => toggleClass(cls)}
                                        className="custom-checkbox mr-2"
                                    />
                                    <label htmlFor={`class-${index}`}>{cls}</label>
                                </div>
                            ))}
                        </div>
                        <div className="flex justify-end p-2">
                            <button
                                className="bg-[#7A8BFF] text-white py-1 px-3 rounded-full"
                                onClick={toggleAllClasses}
                            >
                                {selectedClasses.length === options.length
                                    ? "Deselect All"
                                    : "Select All"}
                            </button>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

export default SelectCheckbox;
