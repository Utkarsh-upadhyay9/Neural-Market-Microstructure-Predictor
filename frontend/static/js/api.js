/**
 * Neural Market Predictor - API Client
 * Author: Utkarsh Upadhyay (@Utkarsh-upadhyay9)
 */

class APIClient {
    constructor(baseURL = '') {
        this.baseURL = baseURL;
    }

    async request(endpoint, options = {}) {
        try {
            const url = `${this.baseURL}/api${endpoint}`;
            const response = await fetch(url, {
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                ...options
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    }

    async getPredictions(params = {}) {
        const queryString = new URLSearchParams(params).toString();
        const endpoint = queryString ? `/predictions?${queryString}` : '/predictions';
        return this.request(endpoint);
    }

    async getSinglePrediction(symbol) {
        return this.request(`/predict/${symbol}`);
    }

    async getSymbols() {
        return this.request('/symbols');
    }

    async getModelStatus() {
        return this.request('/model/status');
    }

    async refreshPredictions(symbols = null) {
        const params = symbols ? `?symbols=${symbols.join(',')}` : '';
        return this.request(`/refresh${params}`);
    }

    async getPopular() {
        return this.request('/popular');
    }
}

// Export for use in other modules
window.APIClient = APIClient;
