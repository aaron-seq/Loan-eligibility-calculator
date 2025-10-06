/**
 * Common JavaScript utilities for the Loan Eligibility System
 */

// Show flash messages
function showMessage(message, type = 'info') {
    const colors = {
        success: 'bg-green-100 border-green-400 text-green-700',
        error: 'bg-red-100 border-red-400 text-red-700',
        warning: 'bg-yellow-100 border-yellow-400 text-yellow-700',
        info: 'bg-blue-100 border-blue-400 text-blue-700'
    };

    const icons = {
        success: 'fas fa-check-circle',
        error: 'fas fa-exclamation-circle',
        warning: 'fas fa-exclamation-triangle',
        info: 'fas fa-info-circle'
    };

    const flashContainer = document.getElementById('flash-messages');
    const messageDiv = document.createElement('div');
    
    messageDiv.className = `border-l-4 p-4 ${colors[type]} rounded-md mb-4`;
    messageDiv.innerHTML = `
        <div class="flex">
            <div class="flex-shrink-0">
                <i class="${icons[type]}"></i>
            </div>
            <div class="ml-3">
                <p class="text-sm">${message}</p>
            </div>
            <div class="ml-auto pl-3">
                <div class="-mx-1.5 -my-1.5">
                    <button onclick="this.parentElement.parentElement.parentElement.remove()"
                            class="inline-flex rounded-md p-1.5 focus:outline-none focus:ring-2 focus:ring-offset-2 hover:opacity-75">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            </div>
        </div>
    `;

    flashContainer.appendChild(messageDiv);

    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (messageDiv.parentNode) {
            messageDiv.remove();
        }
    }, 5000);
}

// Form validation utilities
function validateForm(formId, validationRules) {
    const form = document.getElementById(formId);
    if (!form) return false;

    let isValid = true;
    const formData = new FormData(form);

    for (const [field, rules] of Object.entries(validationRules)) {
        const value = formData.get(field);
        const fieldElement = document.getElementById(field);

        // Reset previous validation state
        fieldElement.classList.remove('border-red-500');
        
        // Remove existing error message
        const existingError = fieldElement.parentNode.querySelector('.error-message');
        if (existingError) {
            existingError.remove();
        }

        for (const rule of rules) {
            if (!rule.validator(value)) {
                isValid = false;
                fieldElement.classList.add('border-red-500');
                
                // Add error message
                const errorDiv = document.createElement('div');
                errorDiv.className = 'error-message text-sm text-red-600 mt-1';
                errorDiv.textContent = rule.message;
                fieldElement.parentNode.appendChild(errorDiv);
                break;
            }
        }
    }

    return isValid;
}

// Loading state utilities
function setLoadingState(buttonElement, isLoading, loadingText = 'Processing...') {
    if (isLoading) {
        buttonElement.disabled = true;
        buttonElement.innerHTML = `<i class="fas fa-spinner fa-spin mr-2"></i>${loadingText}`;
    } else {
        buttonElement.disabled = false;
        // Restore original button text
        const originalText = buttonElement.getAttribute('data-original-text') || 'Submit';
        buttonElement.innerHTML = originalText;
    }
}

// API request helper
async function makeApiRequest(url, options = {}) {
    try {
        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });

        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.message || 'Request failed');
        }

        return data;
    } catch (error) {
        console.error('API request failed:', error);
        throw error;
    }
}

// Initialize common functionality on page load
document.addEventListener('DOMContentLoaded', function() {
    // Store original button texts
    document.querySelectorAll('button[type="submit"]').forEach(button => {
        button.setAttribute('data-original-text', button.textContent.trim());
    });

    // Add smooth scrolling to anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });
});
