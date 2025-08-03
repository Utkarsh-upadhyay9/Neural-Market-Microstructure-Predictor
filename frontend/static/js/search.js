/**
 * Neural Market Predictor - Enhanced Search with Immediate Display
 * Author: Utkarsh Upadhyay (@Utkarsh-upadhyay9)
 */

class StockSearch {
    constructor(apiClient) {
        this.api = apiClient;
        this.stockUniverse = [];
        this.searchTimeout = null;
        this.selectedStocks = new Set();
        this.watchlist = new Set(this.loadWatchlist());
        
        this.init();
    }

    async init() {
        await this.loadStockUniverse();
        this.bindEvents();
        this.updateWatchlistDisplay();
    }

    async loadStockUniverse() {
        try {
            const response = await this.api.request('/symbols');
            if (response.success) {
                this.stockUniverse = this.formatStockUniverse(response.data);
                console.log(`Loaded ${this.stockUniverse.length} stocks for search`);
            }
        } catch (error) {
            console.error('Failed to load stock universe:', error);
        }
    }

    formatStockUniverse(data) {
        const stocks = [];
        
        // Process all symbols with sector information
        Object.entries(data.by_sector).forEach(([sector, symbols]) => {
            symbols.forEach(symbol => {
                const config = this.getStockInfo(symbol);
                stocks.push({
                    symbol: symbol,
                    name: config.name,
                    sector: sector,
                    searchText: `${symbol.toLowerCase()} ${config.name.toLowerCase()}`
                });
            });
        });

        return stocks.sort((a, b) => a.symbol.localeCompare(b.symbol));
    }

    getStockInfo(symbol) {
        // Enhanced stock information mapping
        const stockInfo = {
            'AAPL': { name: 'Apple Inc.' },
            'GOOGL': { name: 'Alphabet Inc.' },
            'MSFT': { name: 'Microsoft Corporation' },
            'AMZN': { name: 'Amazon.com Inc.' },
            'TSLA': { name: 'Tesla Inc.' },
            'META': { name: 'Meta Platforms Inc.' },
            'NVDA': { name: 'NVIDIA Corporation' },
            'NFLX': { name: 'Netflix Inc.' },
            'JPM': { name: 'JPMorgan Chase & Co.' },
            'JNJ': { name: 'Johnson & Johnson' },
            'PG': { name: 'Procter & Gamble Co.' },
            'HD': { name: 'Home Depot Inc.' },
            'BAC': { name: 'Bank of America Corp.' },
            'XOM': { name: 'Exxon Mobil Corporation' },
            'CVX': { name: 'Chevron Corporation' },
            'PFE': { name: 'Pfizer Inc.' },
            'KO': { name: 'Coca-Cola Company' },
            'PEP': { name: 'PepsiCo Inc.' },
            'SPY': { name: 'SPDR S&P 500 ETF Trust' },
            'QQQ': { name: 'Invesco QQQ Trust' }
        };
        
        return stockInfo[symbol] || { name: `${symbol} Corporation` };
    }

    bindEvents() {
        const searchInput = document.getElementById('stock-search');
        const searchResults = document.getElementById('search-results');

        if (!searchInput) return;

        searchInput.addEventListener('input', (e) => {
            const query = e.target.value.trim();
            
            if (this.searchTimeout) {
                clearTimeout(this.searchTimeout);
            }

            this.searchTimeout = setTimeout(() => {
                if (query.length >= 1) {
                    this.performSearch(query);
                } else {
                    this.hideSearchResults();
                }
            }, 100); // Faster search response
        });

        searchInput.addEventListener('focus', () => {
            if (searchInput.value.trim().length >= 1) {
                this.performSearch(searchInput.value.trim());
            }
        });

        // Hide search results when clicking outside
        document.addEventListener('click', (e) => {
            if (!searchInput.contains(e.target) && !searchResults.contains(e.target)) {
                this.hideSearchResults();
            }
        });

        // Handle enter key
        searchInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                const firstResult = searchResults.querySelector('.search-result-item');
                if (firstResult) {
                    firstResult.click();
                }
            } else if (e.key === 'Escape') {
                this.hideSearchResults();
                searchInput.blur();
            }
        });
    }

    performSearch(query) {
        const lowerQuery = query.toLowerCase();
        const results = this.stockUniverse.filter(stock => 
            stock.searchText.includes(lowerQuery) ||
            stock.symbol.toLowerCase().startsWith(lowerQuery)
        ).slice(0, 15); // Show more results

        this.displaySearchResults(results, query);
    }

    displaySearchResults(results, query) {
        const searchResults = document.getElementById('search-results');
        if (!searchResults) return;

        if (results.length === 0) {
            searchResults.innerHTML = `
                <div class="search-result-item no-results">
                    <div class="result-main">
                        <div class="result-symbol">No stocks found for "${query}"</div>
                        <div class="result-name">Try searching by symbol (e.g., AAPL) or company name</div>
                    </div>
                </div>
            `;
        } else {
            searchResults.innerHTML = results.map(stock => `
                <div class="search-result-item" data-symbol="${stock.symbol}">
                    <div class="result-main">
                        <div class="result-symbol">${stock.symbol}</div>
                        <div class="result-name">${stock.name}</div>
                    </div>
                    <div class="result-sector">${stock.sector}</div>
                </div>
            `).join('');

            // Bind click events
            searchResults.querySelectorAll('.search-result-item').forEach(item => {
                const symbol = item.dataset.symbol;
                if (symbol) {
                    item.addEventListener('click', () => this.selectStock(symbol));
                    
                    // Add hover effect
                    item.addEventListener('mouseenter', () => {
                        item.style.backgroundColor = 'var(--neutral-50)';
                    });
                    item.addEventListener('mouseleave', () => {
                        item.style.backgroundColor = '';
                    });
                }
            });
        }

        searchResults.style.display = 'block';
    }

    hideSearchResults() {
        const searchResults = document.getElementById('search-results');
        if (searchResults) {
            searchResults.style.display = 'none';
        }
    }

    async selectStock(symbol) {
        console.log(`Stock selected: ${symbol}`);
        
        // Clear search
        const searchInput = document.getElementById('stock-search');
        if (searchInput) {
            searchInput.value = '';
        }
        this.hideSearchResults();

        // Add to selected stocks
        this.selectedStocks.add(symbol);
        
        // Immediately trigger prediction and display
        if (window.dashboard) {
            console.log(`Triggering live stock addition for: ${symbol}`);
            await window.dashboard.addLiveStock(symbol);
        } else {
            console.error('Dashboard not available');
        }

        // Show user feedback
        this.showSelectionFeedback(symbol);
    }

    showSelectionFeedback(symbol) {
        // Create a temporary notification
        const notification = document.createElement('div');
        notification.className = 'stock-selected-notification';
        notification.innerHTML = `
            <div class="notification-content">
                <span class="notification-icon">✓</span>
                <span class="notification-text">Added ${symbol} to live predictions</span>
            </div>
        `;
        
        // Add styles
        notification.style.cssText = `
            position: fixed;
            top: 80px;
            right: 20px;
            background: var(--success);
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            box-shadow: var(--shadow-lg);
            z-index: 1002;
            animation: slideIn 0.3s ease-out;
        `;

        document.body.appendChild(notification);

        // Remove after 3 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease-in forwards';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 3000);
    }

    addToWatchlist(symbol) {
        this.watchlist.add(symbol);
        this.saveWatchlist();
        this.updateWatchlistDisplay();
    }

    removeFromWatchlist(symbol) {
        this.watchlist.delete(symbol);
        this.saveWatchlist();
        this.updateWatchlistDisplay();
    }

    isInWatchlist(symbol) {
        return this.watchlist.has(symbol);
    }

    clearWatchlist() {
        this.watchlist.clear();
        this.saveWatchlist();
        this.updateWatchlistDisplay();
    }

    updateWatchlistDisplay() {
        const watchlistSection = document.getElementById('watchlist-section');
        const watchlistGrid = document.getElementById('watchlist-grid');
        
        if (!watchlistSection || !watchlistGrid) return;

        if (this.watchlist.size === 0) {
            watchlistSection.style.display = 'none';
        } else {
            watchlistSection.style.display = 'block';
        }
    }

    loadWatchlist() {
        try {
            const saved = localStorage.getItem('neural-predictor-watchlist');
            return saved ? JSON.parse(saved) : [];
        } catch (error) {
            console.error('Failed to load watchlist:', error);
            return [];
        }
    }

    saveWatchlist() {
        try {
            localStorage.setItem('neural-predictor-watchlist', 
                JSON.stringify(Array.from(this.watchlist)));
        } catch (error) {
            console.error('Failed to save watchlist:', error);
        }
    }

    getSelectedStocks() {
        return Array.from(this.selectedStocks);
    }

    getWatchlist() {
        return Array.from(this.watchlist);
    }
}

// Add notification animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
    
    .notification-content {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .notification-icon {
        font-size: 16px;
        font-weight: bold;
    }
    
    .notification-text {
        font-size: 14px;
        font-weight: 500;
    }
    
    .search-result-item.no-results {
        cursor: default;
        opacity: 0.7;
    }
    
    .search-result-item.no-results:hover {
        background: none !important;
    }
`;
document.head.appendChild(style);

// Export for use in dashboard
window.StockSearch = StockSearch;
