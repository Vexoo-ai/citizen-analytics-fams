// CitizenAnalytics™ Model Selection Frontend Application
class CitizenAnalyticsApp {
    constructor() {
        this.currentStep = 1;
        this.fileId = null;
        this.jobId = null;
        this.previewData = null;
        this.categoricalConfigs = [];
        this.targetVisualization = null;
        this.adasynRecommendation = null;
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.updateStepIndicator();
    }

    setupEventListeners() {
        // File upload
        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('file-input');
        
        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
        uploadArea.addEventListener('drop', this.handleDrop.bind(this));
        fileInput.addEventListener('change', this.handleFileSelect.bind(this));

        // Navigation buttons
        document.getElementById('configure-btn').addEventListener('click', () => this.goToStep(3));
        document.getElementById('back-to-preview').addEventListener('click', () => this.goToStep(2));
        document.getElementById('visualize-btn').addEventListener('click', () => this.visualizeTarget());
        document.getElementById('back-to-configure').addEventListener('click', () => this.goToStep(3));
        document.getElementById('start-analysis-btn').addEventListener('click', () => this.goToStep(5));
        document.getElementById('back-to-visualize').addEventListener('click', () => this.goToStep(4));
        document.getElementById('run-analysis-btn').addEventListener('click', () => this.runAnalysis());
        document.getElementById('new-analysis-btn').addEventListener('click', () => this.resetApp());
        document.getElementById('cleanup-btn').addEventListener('click', () => this.cleanupJob());

        // Tab navigation
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.switchTab(e.target.dataset.tab));
        });

        // Problem type change
        document.getElementById('problem-type').addEventListener('change', this.updateMetricOptions.bind(this));
    }

    // Drag and Drop Handlers
    handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        e.currentTarget.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        e.currentTarget.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        e.currentTarget.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.uploadFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.uploadFile(file);
        }
    }

    // File Upload
    async uploadFile(file) {
        try {
            this.showLoading('Uploading file...');
            this.showUploadProgress();

            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Upload failed: ${response.statusText}`);
            }

            const result = await response.json();
            this.fileId = result.file_id;
            
            this.hideLoading();
            this.hideUploadProgress();
            this.showToast('File uploaded successfully!', 'success');
            
            // Load preview data
            await this.loadPreviewData();
            this.goToStep(2);

        } catch (error) {
            this.hideLoading();
            this.hideUploadProgress();
            this.showToast(`Upload failed: ${error.message}`, 'error');
        }
    }

    // Load Preview Data
    async loadPreviewData() {
        try {
            const response = await fetch(`/preview/${this.fileId}`);
            if (!response.ok) {
                throw new Error(`Preview failed: ${response.statusText}`);
            }

            this.previewData = await response.json();
            this.renderPreviewData();
            this.populateTargetSelect();
            this.initializeCategoricalConfigs();

        } catch (error) {
            this.showToast(`Failed to load preview: ${error.message}`, 'error');
        }
    }

    // Render Preview Data
    renderPreviewData() {
        const data = this.previewData;
        
        // Data summary
        const summaryHtml = `
            <div class="summary-item">
                <h4>Rows</h4>
                <div class="value">${data.rows.toLocaleString()}</div>
            </div>
            <div class="summary-item">
                <h4>Columns</h4>
                <div class="value">${data.columns.length}</div>
            </div>
            <div class="summary-item">
                <h4>Missing Values</h4>
                <div class="value">${Object.values(data.missing_values).reduce((a, b) => a + b, 0)}</div>
            </div>
            <div class="summary-item">
                <h4>File Size</h4>
                <div class="value">${this.formatFileSize(data.file_size || 0)}</div>
            </div>
        `;
        document.getElementById('data-summary').innerHTML = summaryHtml;

        // Target suggestions
        const targetHtml = data.suggested_target_columns.map(col => 
            `<span class="suggestion-tag">${col}</span>`
        ).join('');
        document.getElementById('target-suggestions').innerHTML = targetHtml || '<em>No suggestions</em>';

        // Categorical suggestions
        const categoricalHtml = data.categorical_suggestions.map(item => 
            `<span class="suggestion-tag ${item.suggested_type}" title="${item.reasoning}">
                ${item.variable_name} (${item.suggested_type})
            </span>`
        ).join('');
        document.getElementById('categorical-suggestions').innerHTML = categoricalHtml || '<em>No categorical variables</em>';

        // Remove suggestions
        const removeHtml = data.suggested_remove_columns.map(col => 
            `<span class="suggestion-tag">${col}</span>`
        ).join('');
        document.getElementById('remove-suggestions').innerHTML = removeHtml || '<em>No suggestions</em>';

        // Data table
        this.renderDataTable();
    }

    renderDataTable() {
        const data = this.previewData;
        if (!data.preview_data || data.preview_data.length === 0) return;

        const headers = data.columns;
        const rows = data.preview_data;

        let tableHtml = '<thead><tr>';
        headers.forEach(header => {
            const dataType = data.data_types[header];
            tableHtml += `<th>${header} <small>(${dataType})</small></th>`;
        });
        tableHtml += '</tr></thead><tbody>';

        rows.forEach(row => {
            tableHtml += '<tr>';
            headers.forEach(header => {
                const value = row[header];
                tableHtml += `<td>${value !== null && value !== undefined ? value : '<em>null</em>'}</td>`;
            });
            tableHtml += '</tr>';
        });
        tableHtml += '</tbody>';

        document.getElementById('data-table').innerHTML = tableHtml;
    }

    // Populate Target Select
    populateTargetSelect() {
        const select = document.getElementById('target-select');
        select.innerHTML = '<option value="">Select target column...</option>';
        
        this.previewData.columns.forEach(col => {
            const option = document.createElement('option');
            option.value = col;
            option.textContent = col;
            if (this.previewData.suggested_target_columns.includes(col)) {
                option.textContent += ' (suggested)';
                option.selected = true;
            }
            select.appendChild(option);
        });
    }

    // Initialize Categorical Configurations
    initializeCategoricalConfigs() {
        this.categoricalConfigs = this.previewData.categorical_suggestions.map(item => ({
            variable_name: item.variable_name,
            variable_type: item.suggested_type,
            value_ordering: item.suggested_type === 'ordinal' ? item.unique_values : null,
            unique_values: item.unique_values
        }));
        
        this.renderCategoricalConfig();
    }

    renderCategoricalConfig() {
        const container = document.getElementById('categorical-config');
        
        if (this.categoricalConfigs.length === 0) {
            container.innerHTML = '<p>No categorical variables found.</p>';
            return;
        }

        let html = '<h3>Categorical Variables Configuration</h3>';
        
        this.categoricalConfigs.forEach((config, index) => {
            html += `
                <div class="categorical-item">
                    <div class="categorical-header">
                        <div class="categorical-name">${config.variable_name}</div>
                        <div class="type-selector">
                            <button type="button" class="type-btn ${config.variable_type === 'ordinal' ? 'active' : ''}" 
                                    onclick="app.setCategoricalType(${index}, 'ordinal')">Ordinal</button>
                            <button type="button" class="type-btn ${config.variable_type === 'nominal' ? 'active' : ''}" 
                                    onclick="app.setCategoricalType(${index}, 'nominal')">Nominal</button>
                        </div>
                    </div>
                    <div class="ordinal-config ${config.variable_type === 'ordinal' ? 'active' : ''}" id="ordinal-${index}">
                        <label>Value Ordering (drag to reorder):</label>
                        <div class="value-list" id="values-${index}">
                            ${(config.value_ordering || config.unique_values).map((value, valueIndex) => 
                                `<div class="value-item" draggable="true" data-index="${valueIndex}" data-config="${index}">${value}</div>`
                            ).join('')}
                        </div>
                    </div>
                </div>
            `;
        });
        
        container.innerHTML = html;
        this.setupDragAndDrop();
    }

    setCategoricalType(configIndex, type) {
        this.categoricalConfigs[configIndex].variable_type = type;
        
        // Update UI
        const item = document.querySelectorAll('.categorical-item')[configIndex];
        const buttons = item.querySelectorAll('.type-btn');
        const ordinalConfig = item.querySelector('.ordinal-config');
        
        buttons.forEach(btn => btn.classList.remove('active'));
        item.querySelector(`[onclick*="'${type}'"]`).classList.add('active');
        
        if (type === 'ordinal') {
            ordinalConfig.classList.add('active');
        } else {
            ordinalConfig.classList.remove('active');
            this.categoricalConfigs[configIndex].value_ordering = null;
        }
    }

    setupDragAndDrop() {
        document.querySelectorAll('.value-item').forEach(item => {
            item.addEventListener('dragstart', this.handleDragStart.bind(this));
            item.addEventListener('dragover', this.handleDragOverValue.bind(this));
            item.addEventListener('drop', this.handleDropValue.bind(this));
            item.addEventListener('dragend', this.handleDragEnd.bind(this));
        });
    }

    handleDragStart(e) {
        e.dataTransfer.setData('text/plain', '');
        e.target.classList.add('dragging');
        this.draggedElement = e.target;
    }

    handleDragOverValue(e) {
        e.preventDefault();
    }

    handleDropValue(e) {
        e.preventDefault();
        if (this.draggedElement && e.target !== this.draggedElement) {
            const configIndex = parseInt(this.draggedElement.dataset.config);
            const draggedIndex = parseInt(this.draggedElement.dataset.index);
            const targetIndex = parseInt(e.target.dataset.index);
            
            if (configIndex === parseInt(e.target.dataset.config)) {
                this.reorderValues(configIndex, draggedIndex, targetIndex);
            }
        }
    }

    handleDragEnd(e) {
        e.target.classList.remove('dragging');
        this.draggedElement = null;
    }

    reorderValues(configIndex, fromIndex, toIndex) {
        const config = this.categoricalConfigs[configIndex];
        const values = [...config.value_ordering];
        const [movedValue] = values.splice(fromIndex, 1);
        values.splice(toIndex, 0, movedValue);
        
        config.value_ordering = values;
        this.renderCategoricalConfig();
    }

    // Target Visualization
    async visualizeTarget() {
        const targetColumn = document.getElementById('target-select').value;
        const problemType = document.getElementById('problem-type').value;
        
        if (!targetColumn) {
            this.showToast('Please select a target column', 'warning');
            return;
        }

        try {
            this.showLoading('Generating target visualization...');
            
            // Save categorical configurations
            await this.saveCategoricalConfigs();
            
            // Get target visualization
            const response = await fetch(`/visualize-target/${this.fileId}?target_column=${targetColumn}&problem_type=${problemType}`);
            if (!response.ok) {
                throw new Error(`Visualization failed: ${response.statusText}`);
            }

            this.targetVisualization = await response.json();
            
            // Get ADASYN recommendation
            try {
                const adasynResponse = await fetch(`/adasyn-recommendation/${this.fileId}?target_column=${targetColumn}&problem_type=${problemType}`);
                if (adasynResponse.ok) {
                    this.adasynRecommendation = await adasynResponse.json();
                } else {
                    console.warn('ADASYN recommendation failed, using default');
                    this.adasynRecommendation = {
                        recommended: false,
                        reason: "Unable to determine ADASYN recommendation",
                        class_imbalance_info: null
                    };
                }
            } catch (error) {
                console.warn('ADASYN recommendation error:', error);
                this.adasynRecommendation = {
                    recommended: false,
                    reason: "ADASYN recommendation service unavailable",
                    class_imbalance_info: null
                };
            }
            
            this.hideLoading();
            this.renderTargetVisualization();
            this.goToStep(4);

        } catch (error) {
            this.hideLoading();
            this.showToast(`Visualization failed: ${error.message}`, 'error');
        }
    }

    async saveCategoricalConfigs() {
        const targetColumn = document.getElementById('target-select').value;
        const problemType = document.getElementById('problem-type').value;
        
        const payload = {
            file_id: this.fileId,
            target_column: targetColumn,
            problem_type: problemType,
            categorical_configs: this.categoricalConfigs.map(config => ({
                variable_name: config.variable_name,
                variable_type: config.variable_type,
                value_ordering: config.variable_type === 'ordinal' ? config.value_ordering : null
            }))
        };

        const response = await fetch('/configure-categoricals', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`Configuration failed: ${response.statusText}`);
        }

        const result = await response.json();
        if (result.warnings && result.warnings.length > 0) {
            result.warnings.forEach(warning => {
                this.showToast(warning, 'warning');
            });
        }
    }

    renderTargetVisualization() {
        const viz = this.targetVisualization;
        
        // Display chart
        const chartImg = document.getElementById('target-chart');
        chartImg.src = `data:image/png;base64,${viz.chart_base64}`;
        chartImg.style.display = 'block';
        
        // Display statistics
        const stats = viz.statistics;
        let statsHtml = '';
        
        if (viz.problem_type === 'classification') {
            statsHtml = `
                <div class="stat-item">
                    <span class="stat-label">Total Samples</span>
                    <span class="stat-value">${stats.total_samples.toLocaleString()}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Number of Classes</span>
                    <span class="stat-value">${stats.num_classes}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Most Frequent Class</span>
                    <span class="stat-value">${stats.most_frequent_class}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Least Frequent Class</span>
                    <span class="stat-value">${stats.least_frequent_class}</span>
                </div>
            `;
            
            // Add class percentages
            Object.entries(stats.class_percentages).forEach(([cls, pct]) => {
                statsHtml += `
                    <div class="stat-item">
                        <span class="stat-label">${cls}</span>
                        <span class="stat-value">${pct}%</span>
                    </div>
                `;
            });
        } else {
            statsHtml = `
                <div class="stat-item">
                    <span class="stat-label">Mean</span>
                    <span class="stat-value">${stats.mean.toFixed(3)}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Median</span>
                    <span class="stat-value">${stats.median.toFixed(3)}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Std Dev</span>
                    <span class="stat-value">${stats.std.toFixed(3)}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Min</span>
                    <span class="stat-value">${stats.min.toFixed(3)}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Max</span>
                    <span class="stat-value">${stats.max.toFixed(3)}</span>
                </div>
            `;
        }
        
        document.getElementById('target-stats').innerHTML = statsHtml;
        
        // Display ADASYN recommendation
        this.renderAdasynRecommendation();
    }

    renderAdasynRecommendation() {
        if (!this.adasynRecommendation) return;
        
        const container = document.getElementById('adasyn-recommendation');
        const rec = this.adasynRecommendation;
        
        let className = 'adasyn-recommendation';
        let icon = 'fas fa-info-circle';
        
        if (rec.recommended) {
            className += ' recommended';
            icon = 'fas fa-check-circle';
        } else {
            className += ' not-recommended';
            icon = 'fas fa-times-circle';
        }
        
        let detailsHtml = '';
        if (rec.class_imbalance_info) {
            const info = rec.class_imbalance_info;
            detailsHtml = `
                <div style="margin-top: 1rem; font-size: 0.875rem;">
                    <strong>Class Distribution:</strong><br>
                    ${info.majority_class}: ${info.majority_percentage}% (${info.class_distribution[info.majority_class]} samples)<br>
                    ${info.minority_class}: ${info.minority_percentage}% (${info.class_distribution[info.minority_class]} samples)
                </div>
            `;
        }
        
        container.className = className;
        container.innerHTML = `
            <div class="adasyn-header">
                <i class="${icon}"></i>
                ADASYN ${rec.recommended ? 'Recommended' : 'Not Recommended'}
            </div>
            <div class="adasyn-reason">${rec.reason}</div>
            ${detailsHtml}
        `;
        
        // Set ADASYN checkbox based on recommendation
        document.getElementById('apply-adasyn').checked = rec.recommended;
    }

    // Analysis Configuration
    updateMetricOptions() {
        const problemType = document.getElementById('problem-type').value;
        const metricSelect = document.getElementById('metric-select');
        
        metricSelect.innerHTML = '';
        
        if (problemType === 'classification') {
            ['accuracy', 'precision', 'recall', 'f1'].forEach(metric => {
                const option = document.createElement('option');
                option.value = metric;
                option.textContent = metric.charAt(0).toUpperCase() + metric.slice(1);
                if (metric === 'accuracy') option.selected = true;
                metricSelect.appendChild(option);
            });
        } else {
            ['mae', 'rmse', 'r2'].forEach(metric => {
                const option = document.createElement('option');
                option.value = metric;
                option.textContent = metric.toUpperCase();
                if (metric === 'mae') option.selected = true;
                metricSelect.appendChild(option);
            });
        }
    }

    // Run Analysis
    async runAnalysis() {
        const targetColumn = document.getElementById('target-select').value;
        const problemType = document.getElementById('problem-type').value;
        const metric = document.getElementById('metric-select').value;
        const iterations = parseInt(document.getElementById('iterations').value);
        const imputeMethod = document.getElementById('impute-method').value;
        const applyAdasyn = document.getElementById('apply-adasyn').checked;
        const skipPycaret = document.getElementById('skip-pycaret').checked;
        const skipVexoo = document.getElementById('skip-vexoo').checked;

        if (!targetColumn) {
            this.showToast('Please select a target column', 'warning');
            return;
        }

        try {
            const payload = {
                file_id: this.fileId,
                target_column: targetColumn,
                problem_type: problemType,
                metric: metric,
                iterations: iterations,
                remove_columns: [],
                impute_method: imputeMethod,
                categorical_configs: this.categoricalConfigs.map(config => ({
                    variable_name: config.variable_name,
                    variable_type: config.variable_type,
                    value_ordering: config.variable_type === 'ordinal' ? config.value_ordering : null
                })),
                apply_adasyn: applyAdasyn,
                skip_pycaret: skipPycaret,
                skip_vexoo: skipVexoo
            };

            const response = await fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error(`Analysis failed: ${response.statusText}`);
            }

            const result = await response.json();
            this.jobId = result.job_id;
            
            this.showToast('Analysis started successfully!', 'success');
            this.showAnalysisProgress();
            this.monitorProgress();

        } catch (error) {
            this.showToast(`Analysis failed: ${error.message}`, 'error');
        }
    }

    showAnalysisProgress() {
        document.getElementById('analysis-config').style.display = 'none';
        document.getElementById('analysis-progress').style.display = 'block';
        document.getElementById('run-analysis-btn').disabled = true;
    }

    async monitorProgress() {
        const checkProgress = async () => {
            try {
                const response = await fetch(`/status/${this.jobId}`);
                if (!response.ok) return;

                const status = await response.json();
                
                // Update progress UI
                const progressFill = document.getElementById('analysis-progress-fill');
                const progressPercentage = document.getElementById('progress-percentage');
                const progressStep = document.getElementById('progress-step');
                const progressTime = document.getElementById('progress-time');
                
                progressFill.style.width = `${status.progress || 0}%`;
                progressPercentage.textContent = `${status.progress || 0}%`;
                progressStep.textContent = status.current_step || 'Processing...';
                progressTime.textContent = status.estimated_remaining ? `Remaining: ${status.estimated_remaining}` : '';

                if (status.status === 'completed') {
                    this.showToast('Analysis completed successfully!', 'success');
                    await this.loadResults();
                    this.goToStep(6);
                    return;
                } else if (status.status === 'failed') {
                    this.showToast(`Analysis failed: ${status.message}`, 'error');
                    document.getElementById('run-analysis-btn').disabled = false;
                    return;
                }

                // Continue monitoring
                setTimeout(checkProgress, 2000);

            } catch (error) {
                console.error('Progress check failed:', error);
                setTimeout(checkProgress, 5000);
            }
        };

        checkProgress();
    }

    // Load Results
    async loadResults() {
        try {
            const response = await fetch(`/results/${this.jobId}`);
            if (!response.ok) {
                throw new Error(`Results loading failed: ${response.statusText}`);
            }

            this.results = await response.json();
            this.renderResults();
            await this.loadDownloads();

        } catch (error) {
            this.showToast(`Failed to load results: ${error.message}`, 'error');
        }
    }

    renderResults() {
        const results = this.results;
        const summary = results.summary;
        
        // Results summary
        const summaryHtml = `
            <h3><i class="fas fa-chart-line"></i> Analysis Summary</h3>
            <div class="summary-grid">
                <div class="summary-card">
                    <h4>Best Performance</h4>
                    <div class="value">${summary.best_score.toFixed(4)}</div>
                    <div class="detail">Seed ${summary.best_seed}</div>
                </div>
                <div class="summary-card">
                    <h4>Worst Performance</h4>
                    <div class="value">${summary.worst_score.toFixed(4)}</div>
                    <div class="detail">Seed ${summary.worst_seed}</div>
                </div>
                <div class="summary-card">
                    <h4>Mean Performance</h4>
                    <div class="value">${summary.mean_score.toFixed(4)}</div>
                    <div class="detail">Across ${results.seed_results.length} seeds</div>
                </div>
                <div class="summary-card">
                    <h4>Standard Deviation</h4>
                    <div class="value">${summary.standard_deviation.toFixed(4)}</div>
                    <div class="detail">Stability metric</div>
                </div>
                <div class="summary-card">
                    <h4>Score Range</h4>
                    <div class="value">${(summary.score_range_max - summary.score_range_min).toFixed(4)}</div>
                    <div class="detail">${summary.score_range_min.toFixed(4)} - ${summary.score_range_max.toFixed(4)}</div>
                </div>
                <div class="summary-card">
                    <h4>Execution Time</h4>
                    <div class="value">${Math.round(results.execution_time)}s</div>
                    <div class="detail">Total analysis time</div>
                </div>
            </div>
        `;
        document.getElementById('results-summary').innerHTML = summaryHtml;

        // Summary tab content
        const summaryGridHtml = `
            <div class="summary-card">
                <h4>Configuration</h4>
                <div class="value">${results.analysis_config.problem_type}</div>
                <div class="detail">Target: ${results.analysis_config.target_column}</div>
            </div>
            <div class="summary-card">
                <h4>Metric</h4>
                <div class="value">${results.analysis_config.metric.toUpperCase()}</div>
                <div class="detail">${summary.higher_is_better ? 'Higher is better' : 'Lower is better'}</div>
            </div>
            <div class="summary-card">
                <h4>Iterations</h4>
                <div class="value">${results.analysis_config.iterations}</div>
                <div class="detail">Random seeds tested</div>
            </div>
            <div class="summary-card">
                <h4>ADASYN Applied</h4>
                <div class="value">${results.analysis_config.apply_adasyn ? 'Yes' : 'No'}</div>
                <div class="detail">Class balancing</div>
            </div>
            <div class="summary-card">
                <h4>PyCaret Available</h4>
                <div class="value">${results.pycaret_available ? 'Yes' : 'No'}</div>
                <div class="detail">Model comparison</div>
            </div>
            <div class="summary-card">
                <h4>AI Analysis Available</h4>
                <div class="value">${results.vexoo_analysis_available ? 'Yes' : 'No'}</div>
                <div class="detail">Vexoo insights</div>
            </div>
        `;
        document.getElementById('summary-grid').innerHTML = summaryGridHtml;

        // Seeds table
        this.renderSeedsTable();
    }

    renderSeedsTable() {
        const seeds = this.results.seed_results;
        
        let tableHtml = `
            <thead>
                <tr>
                    <th>Seed</th>
                    <th>Score</th>
                    <th>Rank</th>
                </tr>
            </thead>
            <tbody>
        `;
        
        const sortedSeeds = [...seeds].sort((a, b) => {
            return this.results.summary.higher_is_better ? b.score - a.score : a.score - b.score;
        });
        
        sortedSeeds.forEach((seed, index) => {
            let rowClass = '';
            if (seed.seed === this.results.summary.best_seed) rowClass = 'best-seed';
            else if (seed.seed === this.results.summary.worst_seed) rowClass = 'worst-seed';
            
            tableHtml += `
                <tr class="${rowClass}">
                    <td>${seed.seed}</td>
                    <td>${seed.score.toFixed(4)}</td>
                    <td>#${index + 1}</td>
                </tr>
            `;
        });
        
        tableHtml += '</tbody>';
        document.getElementById('seeds-table').innerHTML = tableHtml;
    }

    async loadDownloads() {
        try {
            const response = await fetch(`/downloads/${this.jobId}`);
            if (!response.ok) return;

            const downloads = await response.json();
            this.renderDownloads(downloads.available_files);
            this.loadCharts();

        } catch (error) {
            console.error('Failed to load downloads:', error);
        }
    }

    renderDownloads(files) {
        let html = '';
        
        files.forEach(file => {
            const iconClass = this.getFileIcon(file.file_type);
            html += `
                <div class="download-item">
                    <div class="download-info">
                        <div class="download-icon ${file.file_type.toLowerCase()}">${iconClass}</div>
                        <div class="download-details">
                            <h5>${file.filename}</h5>
                            <p>${file.file_type} • ${this.formatFileSize(file.file_size)}</p>
                        </div>
                    </div>
                    <div class="download-actions">
                        <a href="${file.download_url}" class="btn btn-primary btn-sm" download>
                            <i class="fas fa-download"></i> Download
                        </a>
                    </div>
                </div>
            `;
        });
        
        document.getElementById('downloads-list').innerHTML = html;
    }

    loadCharts() {
        // Load line chart
        const lineChart = document.getElementById('line-chart');
        lineChart.src = `/files/${this.jobId}/bias_variance_line_plot.png`;
        lineChart.onerror = () => lineChart.style.display = 'none';
        
        // Load histogram
        const histogramChart = document.getElementById('histogram-chart');
        histogramChart.src = `/files/${this.jobId}/score_distribution.png`;
        histogramChart.onerror = () => histogramChart.style.display = 'none';
    }

    getFileIcon(fileType) {
        const icons = {
            'PNG': '<i class="fas fa-image"></i>',
            'CSV': '<i class="fas fa-file-csv"></i>',
            'MD': '<i class="fas fa-file-alt"></i>',
            'TXT': '<i class="fas fa-file-alt"></i>'
        };
        return icons[fileType] || '<i class="fas fa-file"></i>';
    }

    // Tab Management
    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
        
        // Update tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`tab-${tabName}`).classList.add('active');
        
        // Load specific content if needed
        if (tabName === 'pycaret' && this.results.pycaret_available) {
            this.loadPyCaretResults();
        } else if (tabName === 'vexoo' && this.results.vexoo_analysis_available) {
            this.loadVexooAnalysis();
        }
    }

    async loadPyCaretResults() {
        const container = document.getElementById('pycaret-results');
        if (container.innerHTML.trim()) return; // Already loaded
        
        try {
            const seeds = ['best', 'worst', 'most_common'];
            let html = '';
            
            for (const seed of seeds) {
                const response = await fetch(`/files/${this.jobId}/pycaret_${seed}_seed.csv`);
                if (response.ok) {
                    const csvText = await response.text();
                    const data = this.parseCSV(csvText);
                    
                    html += `
                        <div class="pycaret-section">
                            <h4>${seed.charAt(0).toUpperCase() + seed.slice(1).replace('_', ' ')} Seed Results</h4>
                            ${this.renderPyCaretTable(data)}
                        </div>
                    `;
                }
            }
            
            container.innerHTML = html || '<p>PyCaret results not available.</p>';
            
        } catch (error) {
            container.innerHTML = '<p>Failed to load PyCaret results.</p>';
        }
    }

    parseCSV(csvText) {
        const lines = csvText.trim().split('\n');
        const headers = lines[0].split(',');
        const rows = lines.slice(1, 6).map(line => { // Top 5 only
            const values = line.split(',');
            const row = {};
            headers.forEach((header, index) => {
                row[header] = values[index];
            });
            return row;
        });
        return { headers, rows };
    }

    renderPyCaretTable(data) {
        let html = '<div class="table-wrapper"><table class="data-table"><thead><tr>';
        
        data.headers.forEach(header => {
            html += `<th>${header}</th>`;
        });
        html += '</tr></thead><tbody>';
        
        data.rows.forEach(row => {
            html += '<tr>';
            data.headers.forEach(header => {
                const value = row[header];
                const numValue = parseFloat(value);
                const displayValue = !isNaN(numValue) && header !== 'Model' ? numValue.toFixed(4) : value;
                html += `<td>${displayValue}</td>`;
            });
            html += '</tr>';
        });
        
        html += '</tbody></table></div>';
        return html;
    }

    async loadVexooAnalysis() {
        const container = document.getElementById('vexoo-analysis');
        if (container.innerHTML.trim()) return; // Already loaded
        
        try {
            // Show loading state
            container.innerHTML = '<div style="text-align: center; padding: 2rem;"><i class="fas fa-spinner fa-spin"></i> Loading AI analysis...</div>';
            
            const response = await fetch(`/files/${this.jobId}/vexoo_analysis.md`);
            if (response.ok) {
                const markdown = await response.text();
                
                // Check if we actually got content
                if (markdown.trim()) {
                    // Simple markdown to HTML conversion
                    const html = this.markdownToHtml(markdown);
                    container.innerHTML = `
                        <div class="vexoo-analysis">
                            <h3><i class="fas fa-robot"></i> Vexoo AI Analysis</h3>
                            <div class="vexoo-content">${html}</div>
                        </div>
                    `;
                } else {
                    container.innerHTML = '<div class="vexoo-analysis"><p>Vexoo analysis file is empty.</p></div>';
                }
            } else {
                // Try alternative approach - check if analysis is in results
                if (this.results && this.results.vexoo_analysis_available) {
                    container.innerHTML = '<div class="vexoo-analysis"><p>Vexoo analysis is available but the file could not be loaded. Check the Downloads tab for the markdown file.</p></div>';
                } else {
                    container.innerHTML = '<div class="vexoo-analysis"><p>Vexoo analysis not available. This may happen if the VEXOO_API_KEY environment variable is not set.</p></div>';
                }
            }
        } catch (error) {
            console.error('Vexoo analysis loading error:', error);
            container.innerHTML = `
                <div class="vexoo-analysis">
                    <p>Failed to load Vexoo analysis: ${error.message}</p>
                    <p>You can try downloading the analysis file from the Downloads tab.</p>
                </div>
            `;
        }
    }

    markdownToHtml(markdown) {
        // Enhanced markdown to HTML conversion
        let html = markdown;
        
        // Handle code blocks first to preserve them
        html = html.replace(/```[\s\S]*?```/g, (match) => {
            return `<pre><code>${match.slice(3, -3)}</code></pre>`;
        });
        
        // Handle headers
        html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');
        html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
        html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
        html = html.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
        html = html.replace(/^##### (.+)$/gm, '<h5>$1</h5>');
        
        // Handle bold and italic
        html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
        
        // Handle lists - improved handling
        const lines = html.split('\n');
        let inList = false;
        let processedLines = [];
        
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const isListItem = /^[\s]*[-*] (.+)/.test(line);
            
            if (isListItem && !inList) {
                processedLines.push('<ul>');
                inList = true;
            } else if (!isListItem && inList) {
                processedLines.push('</ul>');
                inList = false;
            }
            
            if (isListItem) {
                const content = line.replace(/^[\s]*[-*] /, '');
                processedLines.push(`<li>${content}</li>`);
            } else {
                processedLines.push(line);
            }
        }
        
        if (inList) {
            processedLines.push('</ul>');
        }
        
        html = processedLines.join('\n');
        
        // Handle paragraphs - but avoid wrapping headers and lists
        html = html.replace(/^(?!<[hul]|<\/[ul]|<li)(.+)$/gm, (match, content) => {
            if (content.trim() === '' || content.includes('<pre>') || content.includes('</pre>')) {
                return match;
            }
            return `<p>${content}</p>`;
        });
        
        // Clean up empty paragraphs
        html = html.replace(/<p><\/p>/g, '');
        html = html.replace(/<p>\s*<\/p>/g, '');
        
        // Handle line breaks
        html = html.replace(/\n\n/g, '</p><p>');
        
        return html;
    }

    // Cleanup
    async cleanupJob() {
        if (!this.jobId) return;
        
        if (!confirm('Are you sure you want to delete all analysis files? This cannot be undone.')) {
            return;
        }
        
        try {
            const response = await fetch(`/cleanup/${this.jobId}`, {
                method: 'DELETE'
            });
            
            if (response.ok) {
                this.showToast('Analysis files cleaned up successfully', 'success');
                this.resetApp();
            } else {
                this.showToast('Cleanup failed', 'error');
            }
        } catch (error) {
            this.showToast(`Cleanup failed: ${error.message}`, 'error');
        }
    }

    // Step Navigation
    goToStep(stepNumber) {
        // Update step indicator
        document.querySelectorAll('.step').forEach((step, index) => {
            step.classList.remove('active', 'completed');
            if (index + 1 < stepNumber) {
                step.classList.add('completed');
            } else if (index + 1 === stepNumber) {
                step.classList.add('active');
            }
        });
        
        // Update step content
        document.querySelectorAll('.step-content').forEach(content => {
            content.classList.remove('active');
        });
        
        const stepMap = {
            1: 'upload-section',
            2: 'preview-section',
            3: 'configure-section',
            4: 'visualize-section',
            5: 'analysis-section',
            6: 'results-section'
        };
        
        document.getElementById(stepMap[stepNumber]).classList.add('active');
        this.currentStep = stepNumber;
        
        // Update metric options when going to analysis step
        if (stepNumber === 5) {
            this.updateMetricOptions();
        }
    }

    updateStepIndicator() {
        this.goToStep(1);
    }

    resetApp() {
        this.currentStep = 1;
        this.fileId = null;
        this.jobId = null;
        this.previewData = null;
        this.categoricalConfigs = [];
        this.targetVisualization = null;
        this.adasynRecommendation = null;
        this.results = null;
        
        // Reset UI
        document.getElementById('file-input').value = '';
        document.getElementById('upload-progress').style.display = 'none';
        document.getElementById('analysis-progress').style.display = 'none';
        document.getElementById('analysis-config').style.display = 'block';
        document.getElementById('run-analysis-btn').disabled = false;
        
        // Clear content
        document.getElementById('data-summary').innerHTML = '';
        document.getElementById('categorical-config').innerHTML = '';
        document.getElementById('target-chart').style.display = 'none';
        document.getElementById('results-summary').innerHTML = '';
        
        this.goToStep(1);
    }

    // Utility Methods
    showUploadProgress() {
        document.getElementById('upload-progress').style.display = 'block';
        // Simulate progress for demo
        let progress = 0;
        const interval = setInterval(() => {
            progress += 10;
            document.getElementById('progress-fill').style.width = `${progress}%`;
            document.getElementById('progress-text').textContent = `Uploading... ${progress}%`;
            
            if (progress >= 100) {
                clearInterval(interval);
            }
        }, 100);
    }

    hideUploadProgress() {
        document.getElementById('upload-progress').style.display = 'none';
        document.getElementById('progress-fill').style.width = '0%';
    }

    showLoading(text = 'Loading...') {
        const overlay = document.getElementById('loading-overlay');
        document.querySelector('.loading-text').textContent = text;
        overlay.style.display = 'flex';
    }

    hideLoading() {
        document.getElementById('loading-overlay').style.display = 'none';
    }

    showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <div>${message}</div>
            <button onclick="this.parentElement.remove()" style="border: none; background: none; float: right; font-size: 1.2em; cursor: pointer;">&times;</button>
        `;
        
        container.appendChild(toast);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (toast.parentElement) {
                toast.remove();
            }
        }, 5000);
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
}

// Initialize the application
const app = new CitizenAnalyticsApp();