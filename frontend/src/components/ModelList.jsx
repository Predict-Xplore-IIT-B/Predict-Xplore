import React from 'react'
import '../styles/admin.css'
import modelImg from '../assets/model.img.png'
import { useState } from 'react';

const ModelList = () => {
    const [searchTerm, setSearchTerm] = useState('');

    const models = [
      {
        id: 1,
        name: 'Model 1',
        description: `Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aenean eu fermentum augue, sit amet convallis augue. Integer eu iaculis sem, sed euismod eros. Nulla facilisi. Proin luctus odio nunc, sed laoreet est bibendum vitae. Sed a eleifend ex. Integer varius rhoncus euismod. Aliquam ac ultrices turpis, vitae eleifend ligula. Aliquam faucibus erat ut tincidunt cursus. Cras at ullamcorper velit. In hac habitasse platea dictumst. Nunc vitae dui.`,
        apiUrl: 'https://www.predictxplore.com/api/v1/test?model=model_id',
        imageUrl: modelImg,
      },
      {
        id: 2,
        name: 'Model 2',
        description: `Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aenean eu fermentum augue, sit amet convallis augue. Integer eu iaculis sem, sed euismod eros. Nulla facilisi. Proin luctus odio nunc, sed laoreet est bibendum vitae. Sed a eleifend ex. Integer varius rhoncus euismod. Aliquam ac ultrices turpis, vitae eleifend ligula. Aliquam faucibus erat ut tincidunt cursus. Cras at ullamcorper velit. In hac habitasse platea dictumst. Nunc vitae dui.`,
        apiUrl: 'https://www.predictxplore.com/api/v1/test?model=model_id',
        imageUrl: modelImg,
      },
    ];

    const filteredModels = models.filter((model) =>
      model.name.toLowerCase().includes(searchTerm.toLowerCase())
    );

    return (
      <div className="admin-model-list">
        {/* Search Bar */}
        <div className="search-bar">
          <input
            type="text"
            placeholder="Search Models ..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>

        {/* Models Container */}
        <div className="outer-container">
          {filteredModels.length > 0 ? (
            filteredModels.map((model) => (
              <div key={model.id} className="model-wrapper">
                <div className="model-card">
                  <div className="model-content">
                    {/* Image */}
                    <img
                      src={model.imageUrl}
                      alt={model.name}
                      className="model-image"
                    />

                    {/* Text Content */}
                    <div className="model-text">
                      <h2>{model.name}</h2>
                      <p>{model.description}</p>
                      <p className="api-url">
                        <strong>API URL: </strong>
                        <a
                          href={model.apiUrl}
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          {model.apiUrl}
                        </a>
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            ))
          ) : (
            <div className="no-results">
              <p>No models found. Try searching with a different keyword.</p>
            </div>
          )}
        </div>
      </div>
    );
  };

  export default ModelList;
