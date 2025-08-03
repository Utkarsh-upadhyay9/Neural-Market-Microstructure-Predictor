/**
 * Neural Market Predictor - Ultra-Fast Live Dashboard
 * Author: Utkarsh Upadhyay (@Utkarsh-upadhyay9)
 * Date: 2025-08-03 16:41:25 UTC
 */

class EnhancedDashboard {
    constructor() {
        this.api = new APIClient();
        this.search = null;
        this.predictions = {};
        this.modelStatus = {};
        this.lastUpdate = null;
        this.autoRefreshInterval = null;
        this.liveUpdateInterval = null;
        this.isRefreshing = false;
        this.isLiveMode = false;
        this.selectedSector = '';
        this.liveStocks = new Set();
        this.searchedStocks = new Set();
        this.updateCounter = 0;
        this.liveUpdateCount = 0;
        
        this.init();
    }

    async init() {
        this.bindEvents();
        this.search = new StockSearch(this.api);
        await this.loadInitialData();
        this.startAutoRefresh();
        
        window.dashboard = this;
    }

    bindEvents() {
        const refreshBtn = document.getElementById('refresh-btn');
        refreshBtn?.addEventListener('click', () => this.handleRefresh());

        const liveToggle = document.getElementById('live-toggle');
        liveToggle?.addEventListener('click', () => this.toggleLiveMode());

        const sectorFilter = document.getElementById('sector-filter');
        sectorFilter?.addEventListener('change', (e) => this.handleSectorChange(e.target.value));

        const clearWatchlist = document.getElementById('clear-watchlist');
        clearWatchlist?.addEventListener('click', () => this.clearWatchlist());

        // Add popular indices button
        const popularBtn = document.createElement('button');
        popularBtn.id = 'popular-btn';
        popularBtn.className = 'popular-btn';
        popularBtn.innerHTML = '⭐ Popular Indices';
        popularBtn.addEventListener('click', () => this.loadPopularIndices());
        
        const controls = document.querySelector('.controls');
        if (controls) {
            controls.appendChild(popularBtn);
        }
    }

    async loadInitialData() {
        await Promise.all([
            this.loadModelStatus(),
            this.loadSectors(),
            this.loadPredictions()
        ]);
    }

    async loadModelStatus() {
        try {
            const response = await this.api.getModelStatus();
            if (response.success) {
                this.modelStatus = response.data;
                this.updateModelStatusUI();
                this.updateConnectionStatus('online');
            }
        } catch (error) {
            console.error('Failed to load model status:', error);
            this.updateConnectionStatus('offline');
        }
    }

    async loadSectors() {
        try {
            const response = await this.api.request('/symbols');
            if (response.success) {
                this.populateSectorFilter(response.data.by_sector);
            }
        } catch (error) {
            console.error('Failed to load sectors:', error);
        }
    }

    async loadPredictions() {
        try {
            this.showLoadingState();
            
            const params = new URLSearchParams();
            if (this.selectedSector) {
                params.append('sector', this.selectedSector);
            }

            if (this.searchedStocks.size > 0) {
                const allSymbols = [...this.searchedStocks];
                if (this.selectedSector) {
                    const sectorStocks = await this.getSectorStocks(this.selectedSector);
                    allSymbols.push(...sectorStocks);
                }
                params.append('symbols', [...new Set(allSymbols)].join(','));
            } else {
                // Load popular stocks by default
                params.append('popular', 'true');
            }

            const response = await this.api.request(`/predictions?${params}`);
            
            if (response.success) {
                this.predictions = response.data;
                this.lastUpdate = response.metadata.last_update;
                this.updateCounter = response.metadata.update_counter || 0;
                this.renderPredictions();
                this.updateLastUpdateTime();
                this.updatePredictionCount();
            }
        } catch (error) {
            console.error('Failed to load predictions:', error);
            this.showErrorState('Failed to load predictions');
        }
    }

    async loadPopularIndices() {
        try {
            const response = await this.api.request('/popular');
            if (response.success) {
                this.predictions = response.data;
                this.renderPredictions();
                this.updatePredictionCount();
                
                // Show notification
                this.showNotification('✅ Popular indices and stocks loaded!');
            }
        } catch (error) {
            console.error('Failed to load popular indices:', error);
        }
    }

    async getSectorStocks(sector) {
        try {
            const response = await this.api.request('/symbols');
            if (response.success) {
                return response.data.by_sector[sector] || [];
            }
        } catch (error) {
            console.error('Failed to get sector stocks:', error);
        }
        return [];
    }

    async addLiveStock(symbol) {
        console.log(`Adding live stock: ${symbol}`);
        this.liveStocks.add(symbol);
        this.searchedStocks.add(symbol);
        
        try {
            const response = await this.api.request(`/predict/${symbol}`);
            if (response.success) {
                this.predictions[symbol] = response.data;
                console.log(`Got prediction for ${symbol}:`, response.data);
                
                this.renderPredictions();
                this.updatePredictionCount();
                
                setTimeout(() => {
                    const newCard = document.querySelector(`[data-symbol="${symbol}"]`);
                    if (newCard) {
                        newCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
                        newCard.style.animation = 'highlight 2s ease-out';
                    }
                }, 100);
                
                this.showNotification(`📊 ${symbol} added to live tracking!`);
            }
        } catch (error) {
            console.error(`Failed to get prediction for ${symbol}:`, error);
        }
    }

    async handleRefresh() {
        if (this.isRefreshing) return;
        
        this.isRefreshing = true;
        const refreshIcon = document.getElementById('refresh-icon');
        refreshIcon?.classList.add('spinning');

        try {
            const params = new URLSearchParams();
            if (this.selectedSector) {
                params.append('sector', this.selectedSector);
            }
            if (this.searchedStocks.size > 0) {
                params.append('symbols', [...this.searchedStocks].join(','));
            }
            
            await this.api.request(`/refresh?${params}`);
            
            setTimeout(() => {
                this.loadPredictions();
                this.isRefreshing = false;
                refreshIcon?.classList.remove('spinning');
            }, 500);
        } catch (error) {
            console.error('Refresh failed:', error);
            this.isRefreshing = false;
            refreshIcon?.classList.remove('spinning');
        }
    }

    toggleLiveMode() {
        this.isLiveMode = !this.isLiveMode;
        const liveToggle = document.getElementById('live-toggle');
        
        if (this.isLiveMode) {
            liveToggle?.classList.add('active');
            this.startLiveUpdates();
            this.showNotification('🔴 Live mode enabled - 100ms updates!');
            console.log('Live mode enabled - 100ms updates');
        } else {
            liveToggle?.classList.remove('active');
            this.stopLiveUpdates();
            this.showNotification('⏸️ Live mode disabled');
            console.log('Live mode disabled');
        }
        this.updateConnectionStatus('online');
    }

    startLiveUpdates() {
        // Ultra-fast updates every 100ms as requested
        this.liveUpdateInterval = setInterval(() => {
            if (this.liveStocks.size > 0 || this.searchedStocks.size > 0 || Object.keys(this.predictions).length > 0) {
                this.updateLiveStocks();
                this.liveUpdateCount++;
            }
        }, 100); // Exactly 100ms as requested
        
        console.log('Live updates started: 100ms interval');
    }

    stopLiveUpdates() {
        if (this.liveUpdateInterval) {
            clearInterval(this.liveUpdateInterval);
            this.liveUpdateInterval = null;
            console.log('Live updates stopped');
        }
    }

    async updateLiveStocks() {
        const stocksToUpdate = new Set([
            ...this.liveStocks, 
            ...this.searchedStocks,
            ...Object.keys(this.predictions)
        ]);
        
        if (stocksToUpdate.size === 0) return;

        try {
            // Update in batches for better performance
            const symbols = [...stocksToUpdate].slice(0, 25); // Limit to prevent overload
            
            // Use Promise.all for concurrent updates
            const updatePromises = symbols.map(symbol => this.updateSingleStock(symbol));
            await Promise.all(updatePromises);
            
            // Update display every few cycles to reduce DOM manipulation
            if (this.liveUpdateCount % 3 === 0) {
                this.updateLiveDisplay();
            }
            
        } catch (error) {
            console.error('Live update batch failed:', error);
        }
    }

    async updateSingleStock(symbol) {
        try {
            const response = await this.api.request(`/predict/${symbol}`);
            if (response.success) {
                const oldPrediction = this.predictions[symbol];
                this.predictions[symbol] = response.data;
                
                // Only update specific card elements to reduce DOM operations
                if (this.liveUpdateCount % 5 === 0) { // Update UI every 5 cycles (500ms)
                    this.updatePredictionCard(symbol, oldPrediction, response.data);
                }
            }
        } catch (error) {
            // Silently handle errors to avoid spam
        }
    }

    updateLiveDisplay() {
        // Batch DOM updates for better performance
        this.updateLastUpdateTime();
        this.updateLiveCounters();
    }

    updateLiveCounters() {
        const statusText = document.getElementById('status-text');
        if (statusText && this.isLiveMode) {
            statusText.textContent = `Live (${this.liveUpdateCount} updates)`;
        }
    }

    updatePredictionCard(symbol, oldPrediction, newPrediction) {
        const card = document.querySelector(`[data-symbol="${symbol}"]`);
        if (!card) return;

        const priceElement = card.querySelector('.current-price');
        if (priceElement) {
            priceElement.textContent = `$${newPrediction.current_price.toFixed(2)}`;
            
            // Flash effect for significant price changes
            if (oldPrediction && Math.abs(oldPrediction.current_price - newPrediction.current_price) > 0.01) {
                priceElement.style.animation = 'priceFlash 0.3s ease-out';
                setTimeout(() => {
                    priceElement.style.animation = '';
                }, 300);
            }
        }

        const changeElement = card.querySelector('.price-change');
        if (changeElement) {
            const changeClass = newPrediction.change_24h >= 0 ? 'positive' : 'negative';
            const changeSign = newPrediction.change_24h >= 0 ? '+' : '';
            changeElement.textContent = `${changeSign}${newPrediction.change_24h.toFixed(2)}% (24h)`;
            changeElement.className = `price-change ${changeClass}`;
        }

        // Update trading recommendation
        const recElement = card.querySelector('.recommendation-badge');
        if (recElement && newPrediction.trading_recommendation) {
            const rec = newPrediction.trading_recommendation.recommendation;
            recElement.textContent = rec.replace('_', ' ');
            recElement.className = `recommendation-badge ${rec.toLowerCase()}`;
        }

        // Update timestamp
        const timeElement = card.querySelector('.card-footer span:last-child');
        if (timeElement) {
            timeElement.textContent = this.formatTime(newPrediction.timestamp);
        }
    }

    handleSectorChange(sector) {
        this.selectedSector = sector;
        this.loadPredictions();
    }

    populateSectorFilter(sectorData) {
        const sectorFilter = document.getElementById('sector-filter');
        if (!sectorFilter) return;

        while (sectorFilter.children.length > 1) {
            sectorFilter.removeChild(sectorFilter.lastChild);
        }

        Object.keys(sectorData).sort().forEach(sector => {
            const option = document.createElement('option');
            option.value = sector;
            option.textContent = `${sector} (${sectorData[sector].length})`;
            sectorFilter.appendChild(option);
        });
    }

    updateModelStatusUI() {
        const elements = {
            'model-type': this.modelStatus.model_type || 'Unknown',
            'model-params': this.modelStatus.parameters || 'Unknown',
            'test-loss': this.modelStatus.test_loss ? 
                `${this.modelStatus.test_loss} (MAE: ${this.modelStatus.test_mae})` : 'Unknown',
            'stock-universe': `${this.modelStatus.supported_stocks || 0} stocks`
        };

        Object.entries(elements).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) element.textContent = value;
        });
    }

    updateConnectionStatus(status) {
        const statusDot = document.getElementById('status-dot');
        const statusText = document.getElementById('status-text');

        if (statusDot && statusText) {
            statusDot.className = `status-dot ${status}`;
            if (this.isLiveMode) {
                statusText.textContent = `Live (${this.liveUpdateCount} updates)`;
            } else {
                statusText.textContent = status === 'online' ? 'Connected' : 'Disconnected';
            }
        }
    }

    updateLastUpdateTime() {
        const element = document.getElementById('last-update');
        if (element) {
            const now = new Date();
            element.textContent = now.toLocaleTimeString();
        }
    }

    updatePredictionCount() {
        const element = document.getElementById('prediction-count');
        if (element) {
            const total = Object.keys(this.predictions).length;
            const searched = this.searchedStocks.size;
            const live = this.liveStocks.size;
            element.textContent = searched > 0 ? 
                `${total} (${searched} searched, ${live} live)` : total.toString();
        }
    }

    showNotification(message) {
        const notification = document.createElement('div');
        notification.className = 'live-notification';
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 80px;
            right: 20px;
            background: #10b981;
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 1000;
            animation: slideIn 0.3s ease-out;
            font-size: 14px;
            font-weight: 500;
        `;

        document.body.appendChild(notification);

        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease-in forwards';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 3000);
    }

    renderPredictions() {
        const container = document.getElementById('predictions-grid');
        if (!container) return;

        if (!this.predictions || Object.keys(this.predictions).length === 0) {
            container.innerHTML = this.createErrorCard('No predictions available. Click "⭐ Popular Indices" or search for stocks!');
            return;
        }

        // Sort: searched first, then popular, then alphabetical
        const sortedPredictions = Object.entries(this.predictions).sort(([symbolA, predA], [symbolB, predB]) => {
            const aSearched = this.searchedStocks.has(symbolA);
            const bSearched = this.searchedStocks.has(symbolB);
            const aPopular = predA.popular || false;
            const bPopular = predB.popular || false;
            
            if (aSearched && !bSearched) return -1;
            if (!aSearched && bSearched) return 1;
            if (aPopular && !bPopular) return -1;
            if (!aPopular && bPopular) return 1;
            return symbolA.localeCompare(symbolB);
        });

        const cards = sortedPredictions.map(([symbol, prediction]) => {
                if (prediction.error) {
                    return this.createErrorCard(prediction.error, symbol);
                }
                return this.createPredictionCard(prediction);
            }).join('');

        container.innerHTML = cards;
        this.bindCardEvents();
        this.renderWatchlist();
    }

    bindCardEvents() {
        document.querySelectorAll('.watchlist-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const symbol = btn.dataset.symbol;
                this.toggleWatchlist(symbol);
            });
        });

        document.querySelectorAll('.prediction-card').forEach(card => {
            const symbol = card.dataset.symbol;
            if (symbol) {
                card.addEventListener('click', () => {
                    this.addLiveStock(symbol);
                });
            }
        });
    }

    toggleWatchlist(symbol) {
        if (this.search.isInWatchlist(symbol)) {
            this.search.removeFromWatchlist(symbol);
        } else {
            this.search.addToWatchlist(symbol);
        }
        this.renderPredictions();
    }

    clearWatchlist() {
        this.search.clearWatchlist();
        this.renderPredictions();
    }

    renderWatchlist() {
        const watchlistGrid = document.getElementById('watchlist-grid');
        const watchlistSection = document.getElementById('watchlist-section');
        
        if (!watchlistGrid || !watchlistSection) return;

        const watchlist = this.search.getWatchlist();
        
        if (watchlist.length === 0) {
            watchlistSection.style.display = 'none';
            return;
        }

        watchlistSection.style.display = 'block';
        
        const watchlistCards = watchlist
            .filter(symbol => this.predictions[symbol])
            .map(symbol => this.createPredictionCard(this.predictions[symbol], true))
            .join('');

        watchlistGrid.innerHTML = watchlistCards;
    }

    createPredictionCard(prediction, isWatchlist = false) {
        const direction = prediction.direction.toLowerCase();
        const confidence = prediction.confidence.toLowerCase();
        const changeClass = prediction.change_24h >= 0 ? 'positive' : 'negative';
        const changeSign = prediction.change_24h >= 0 ? '+' : '';
        const watchlistClass = isWatchlist ? 'watchlist' : '';
        const inWatchlist = this.search.isInWatchlist(prediction.symbol);
        const isSearched = this.searchedStocks.has(prediction.symbol);
        const searchedClass = isSearched ? 'searched' : '';
        const hasTrading = prediction.trading_recommendation ? 'has-trading' : '';
        const isIndex = prediction.sector && prediction.sector.includes('ETF') ? 'index-etf' : '';
        const isPopular = prediction.popular ? 'popular' : '';
        const isLive = prediction.data_source && prediction.data_source.includes('live');

        let tradingSection = '';
        if (prediction.trading_recommendation) {
            const rec = prediction.trading_recommendation;
            const recClass = rec.recommendation.toLowerCase();
            
            tradingSection = `
                <div class="trading-section">
                    <div class="trading-header">
                        <span class="recommendation-badge ${recClass}">${rec.recommendation.replace('_', ' ')}</span>
                        <div>
                            <span class="risk-badge ${rec.risk_level.toLowerCase()}">${rec.risk_level} Risk</span>
                        </div>
                    </div>
                    
                    <div class="trading-details">
                        <div class="trading-metric">
                            <div class="trading-metric-label">Confidence</div>
                            <div class="trading-metric-value">${(rec.confidence * 100).toFixed(1)}%</div>
                        </div>
                        <div class="trading-metric">
                            <div class="trading-metric-label">Position</div>
                            <div class="trading-metric-value">${rec.position_size}</div>
                        </div>
                        <div class="trading-metric">
                            <div class="trading-metric-label">Stop Loss</div>
                            <div class="trading-metric-value">$${rec.stop_loss || 'N/A'}</div>
                        </div>
                        <div class="trading-metric">
                            <div class="trading-metric-label">Take Profit</div>
                            <div class="trading-metric-value">$${rec.take_profit || 'N/A'}</div>
                        </div>
                    </div>
                    
                    ${rec.entry_levels && rec.entry_levels.moderate ? `
                    <div class="entry-levels">
                        <div class="entry-levels-title">Entry Levels:</div>
                        ${rec.entry_levels.aggressive ? `<div class="entry-level"><span>Aggressive:</span><span>$${rec.entry_levels.aggressive}</span></div>` : ''}
                        ${rec.entry_levels.moderate ? `<div class="entry-level"><span>Moderate:</span><span>$${rec.entry_levels.moderate}</span></div>` : ''}
                        ${rec.entry_levels.conservative ? `<div class="entry-level"><span>Conservative:</span><span>$${rec.entry_levels.conservative}</span></div>` : ''}
                    </div>
                    ` : ''}
                </div>
            `;
        }

        return `
            <div class="prediction-card ${direction} ${watchlistClass} ${searchedClass} ${hasTrading} ${isIndex} ${isPopular}" data-symbol="${prediction.symbol}">
                ${isLive ? '<div class="live-indicator-card">LIVE</div>' : ''}
                ${prediction.popular ? '<div class="popular-indicator">⭐</div>' : ''}
                ${prediction.market_hours ? '<div class="market-hours-indicator"></div>' : '<div class="market-hours-indicator closed"></div>'}
                
                <div class="card-header">
                    <div class="symbol-info">
                        <div class="symbol">${prediction.symbol}</div>
                        <div class="company-name">${prediction.name || ''}</div>
                        ${isSearched ? '<div class="searched-indicator">Searched</div>' : ''}
                    </div>
                    <div class="card-actions">
                        <button class="watchlist-btn ${inWatchlist ? 'active' : ''}" 
                                data-symbol="${prediction.symbol}" 
                                title="${inWatchlist ? 'Remove from' : 'Add to'} watchlist">
                            ${inWatchlist ? '★' : '☆'}
                        </button>
                        <span class="confidence-badge ${confidence}">${prediction.confidence}</span>
                    </div>
                </div>
                
                <div class="price-section">
                    <div class="current-price">$${prediction.current_price.toFixed(2)}</div>
                    <div class="price-change ${changeClass}">
                        ${changeSign}${prediction.change_24h.toFixed(2)}% (24h)
                    </div>
                    ${prediction.high && prediction.low ? `
                    <div class="price-range">
                        <small>Range: $${prediction.low.toFixed(2)} - $${prediction.high.toFixed(2)}</small>
                    </div>
                    ` : ''}
                </div>
                
                <div class="prediction-section">
                    <div class="direction ${direction}">${prediction.direction}</div>
                    <div class="signal-strength">Signal: ${prediction.prediction_signal}</div>
                </div>
                
                ${tradingSection}
                
                <div class="card-footer">
                    <span>Vol: ${this.formatVolume(prediction.volume)}</span>
                    <span>${prediction.sector || 'Unknown'}</span>
                    <span>${this.formatTime(prediction.timestamp)}</span>
                </div>
            </div>
        `;
    }

    createErrorCard(error, symbol = 'Error') {
        return `
            <div class="prediction-card error-card">
                <div class="card-header">
                    <span class="symbol">${symbol}</span>
                </div>
                <div class="error-message">${error}</div>
            </div>
        `;
    }

    showLoadingState() {
        const container = document.getElementById('predictions-grid');
        if (container) {
            container.innerHTML = `
                <div class="loading">
                    <div class="loading-spinner"></div>
                    Loading live predictions...
                </div>
            `;
        }
    }

    showErrorState(message) {
        const container = document.getElementById('predictions-grid');
        if (container) {
            container.innerHTML = this.createErrorCard(message);
        }
    }

    formatVolume(volume) {
        if (volume >= 1000000) {
            return `${(volume / 1000000).toFixed(1)}M`;
        } else if (volume >= 1000) {
            return `${(volume / 1000).toFixed(1)}K`;
        }
        return volume.toString();
    }

    formatTime(timestamp) {
        const date = new Date(timestamp);
        return date.toLocaleTimeString('en-US', { 
            hour12: false,
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    }

    startAutoRefresh() {
        // Normal auto-refresh every 5 minutes when not in live mode
        this.autoRefreshInterval = setInterval(() => {
            if (!this.isRefreshing && !this.isLiveMode) {
                this.loadPredictions();
            }
        }, 300000);
    }

    stopAutoRefresh() {
        if (this.autoRefreshInterval) {
            clearInterval(this.autoRefreshInterval);
            this.autoRefreshInterval = null;
        }
    }

    destroy() {
        this.stopAutoRefresh();
        this.stopLiveUpdates();
    }
}

// Initialize enhanced dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new EnhancedDashboard();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (window.dashboard) {
        window.dashboard.destroy();
    }
});
