// 2.....//
document.addEventListener('DOMContentLoaded', function() {
    const scrollWrapper = document.querySelector('.categories-scroll-wrapper');
    const scrollLeftButton = document.querySelector('.scroll-left');
    const scrollRightButton = document.querySelector('.scroll-right');
    const scrollIndicator = document.querySelector('.scroll-indicator');

    let scrollPosition = 0;
    const scrollAmount = 200; // Adjust this value to control scroll distance

    scrollLeftButton.addEventListener('click', function() {
        scrollPosition = Math.max(scrollPosition - scrollAmount, 0);
        scrollWrapper.style.transform = `translateX(-${scrollPosition}px)`;
        updateScrollIndicator();
    });

    scrollRightButton.addEventListener('click', function() {
        const maxScroll = scrollWrapper.scrollWidth - scrollWrapper.clientWidth;
        scrollPosition = Math.min(scrollPosition + scrollAmount, maxScroll);
        scrollWrapper.style.transform = `translateX(-${scrollPosition}px)`;
        updateScrollIndicator();
    });

    function updateScrollIndicator() {
        const maxScroll = scrollWrapper.scrollWidth - scrollWrapper.clientWidth;
        const indicatorWidth = (scrollPosition / maxScroll) * 100;
        scrollIndicator.style.width = `${indicatorWidth}%`;
    }
});



// 4rd section *.................................................................................................../
document.addEventListener('DOMContentLoaded', function () {
    // Initialize countdown timers
    initializeCountdownTimers();

    // Filter products based on category
    const filterButtons = document.querySelectorAll('.filter-btn');
    const productCards = document.querySelectorAll('.product-card');

    filterButtons.forEach((button) => {
        button.addEventListener('click', () => {
            // Remove active class from all buttons
            filterButtons.forEach((btn) => btn.classList.remove('active'));
            // Add active class to the clicked button
            button.classList.add('active');

            // Get the filter value from the button's data-filter attribute
            const filterValue = button.getAttribute('data-filter');

            // Show/hide products based on the filter
            productCards.forEach((card) => {
                const category = card.getAttribute('data-category');
                if (filterValue === 'all' || category === filterValue) {
                    card.style.display = 'block'; // Show the product
                } else {
                    card.style.display = 'none'; // Hide the product
                }
            });
        });
    });

    // Add to Cart functionality
    const addToCartButtons = document.querySelectorAll('.add-to-cart');
    const cartNotification = document.createElement('div');
    cartNotification.className = 'cart-notification';
    document.body.appendChild(cartNotification);

    addToCartButtons.forEach((button) => {
        button.addEventListener('click', () => {
        
            // Show cart icon notification
            cartNotification.innerHTML = '<i class="fa-solid fa-cart-shopping"></i>';
            cartNotification.style.display = 'block';

            // Remove success message after 2 seconds
            setTimeout(() => {
                successMessage.remove();
            }, 2000);

            // Hide cart icon notification after 3 seconds
            setTimeout(() => {
                cartNotification.style.display = 'none';
            }, 3000);
        });
    });
});

document.addEventListener('DOMContentLoaded', function () {
    // Initialize countdown timers
    initializeCountdownTimers();

    // Filter products based on category
    const filterButtons = document.querySelectorAll('.filter-btn');
    const productCards = document.querySelectorAll('.product-card');

    filterButtons.forEach((button) => {
        button.addEventListener('click', () => {
            // Remove active class from all buttons
            filterButtons.forEach((btn) => btn.classList.remove('active'));
            // Add active class to the clicked button
            button.classList.add('active');

            // Get the filter value from the button's data-filter attribute
            const filterValue = button.getAttribute('data-filter');

            // Show/hide products based on the filter
            productCards.forEach((card) => {
                const category = card.getAttribute('data-category');
                if (filterValue === 'all' || category === filterValue) {
                    card.style.display = 'block'; // Show the product
                } else {
                    card.style.display = 'none'; // Hide the product
                }
            });
        });
    });

    // Add to Cart functionality
    const addToCartButtons = document.querySelectorAll('.add-to-cart');
    const cartNotification = document.createElement('div');
    cartNotification.className = 'cart-notification';
    document.body.appendChild(cartNotification);

    // Cart count in the header
    const cartCountElement = document.querySelector('.header-actions .action-btn[data-count]');
    let cartCount = 0;

    addToCartButtons.forEach((button) => {
        button.addEventListener('click', () => {
            // Show success message
            cartNotification.textContent = 'Add card successful';
            cartNotification.style.display = 'block';

            // Hide the message after 2 seconds
            setTimeout(() => {
                cartNotification.style.display = 'none';
            }, 2000);

            // Update cart count
            cartCount++;
            cartCountElement.setAttribute('data-count', cartCount);
            cartCountElement.innerHTML = `
                <i class="fa-solid fa-cart-shopping"></i>
                <span class="label">Cart</span>
                <span class="cart-count">${cartCount}</span>
            `;
        });
    });
});

// Function to initialize countdown timers
function initializeCountdownTimers() {
    const productCards = document.querySelectorAll('.product-card');

    productCards.forEach((card) => {
        const countdown = card.querySelector('.countdown');
        const daysBox = countdown.querySelector('.time-box:nth-child(1) .time-value');
        const hoursBox = countdown.querySelector('.time-box:nth-child(2) .time-value');
        const minutesBox = countdown.querySelector('.time-box:nth-child(3) .time-value');
        const secondsBox = countdown.querySelector('.time-box:nth-child(4) .time-value');

        // Set the target date (e.g., 7 days from now)
        const targetDate = new Date();
        targetDate.setDate(targetDate.getDate() + 7);

        // Update the countdown every second
        const timer = setInterval(() => {
            const now = new Date().getTime();
            const distance = targetDate - now;

            // Calculate days, hours, minutes, and seconds
            const days = Math.floor(distance / (1000 * 60 * 60 * 24));
            const hours = Math.floor((distance % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
            const minutes = Math.floor((distance % (1000 * 60 * 60)) / (1000 * 60));
            const seconds = Math.floor((distance % (1000 * 60)) / 1000);

            // Update the countdown boxes
            daysBox.textContent = days;
            hoursBox.textContent = hours;
            minutesBox.textContent = minutes;
            secondsBox.textContent = seconds;

            // If the countdown is over, clear the interval
            if (distance < 0) {
                clearInterval(timer);
                countdown.innerHTML = '<div class="time-box">EXPIRED</div>';
            }
        }, 1000);
    });
}
document.addEventListener('DOMContentLoaded', function () {
    // Cart functionality
    const cartIcon = document.querySelector('.action-btn[data-count]');
    const cartDrawer = document.querySelector('.cart-drawer');
    const closeCartDrawer = document.querySelector('.close-cart-drawer');
    const cartDrawerBody = document.querySelector('.cart-drawer-body');
    const cartCountElement = document.querySelector('.action-btn[data-count] .label');
    const totalPriceElement = document.querySelector('.total-price');

    let cartItems = [];
    let cartTotal = 0;

    // Open cart drawer
    cartIcon.addEventListener('click', (e) => {
        e.preventDefault();
        cartDrawer.style.transition = 'right 0.3s ease-in-out';
        cartDrawer.classList.add('open');
    });

    // Close cart drawer
    closeCartDrawer.addEventListener('click', () => {
        cartDrawer.style.transition = 'right 0.3s ease-in-out';
        cartDrawer.classList.remove('open');
    });

    // Add to Cart functionality
    const addToCartButtons = document.querySelectorAll('.add-to-cart');
    addToCartButtons.forEach((button) => {
        button.addEventListener('click', () => {
            const productCard = button.closest('.product-card');
            const product = {
                name: productCard.querySelector('.product-title').textContent,
                price: parseFloat(productCard.querySelector('.current-price').textContent.replace('$', '')),
                image: productCard.querySelector('.product-image img').src,
                quantity: 1,
            };

            // Check if the product is already in the cart
            const existingProduct = cartItems.find((item) => item.name === product.name);
            if (existingProduct) {
                existingProduct.quantity++;
            } else {
                cartItems.push(product);
            }

            // Update cart total
            cartTotal += product.price;

            // Update cart UI
            updateCartUI();
        });
    });

    // Update cart UI
    function updateCartUI() {
        // Clear cart drawer body
        cartDrawerBody.innerHTML = '';

        // Add cart items to the drawer
        cartItems.forEach((item) => {
            const cartItem = document.createElement('div');
            cartItem.className = 'cart-item';
            cartItem.innerHTML = `
                <img src="${item.image}" alt="${item.name}">
                <div class="cart-item-details">
                    <h4>${item.name}</h4>
                    <p>$${item.price.toFixed(2)}</p>
                    <p class="quantity">Quantity: ${item.quantity}</p>
                </div>
            `;
            cartDrawerBody.appendChild(cartItem);
        });

        // Update cart total
        totalPriceElement.textContent = `$${cartTotal.toFixed(2)}`;

        // Update cart count
        cartCountElement.textContent = cartItems.length;
    }
});




// 5th section *.................................................................................................../

// Set the date we're counting down to (e.g., 7 days from now)
const countDownDate = new Date().getTime() + (7 * 24 * 60 * 60 * 1000);

// Update the countdown every 1 second
const countdownTimer = setInterval(() => {
    // Get the current date and time
    const now = new Date().getTime();

    // Calculate the remaining time
    const timeRemaining = countDownDate - now;

    // Calculate days, hours, minutes, and seconds
    const days = Math.floor(timeRemaining / (1000 * 60 * 60 * 24));
    const hours = Math.floor((timeRemaining % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
    const minutes = Math.floor((timeRemaining % (1000 * 60 * 60)) / (1000 * 60));
    const seconds = Math.floor((timeRemaining % (1000 * 60)) / 1000);

    // Display the result in the countdown element
    document.getElementById("days").innerText = String(days).padStart(2, "0");
    document.getElementById("hours").innerText = String(hours).padStart(2, "0");
    document.getElementById("minutes").innerText = String(minutes).padStart(2, "0");
    document.getElementById("seconds").innerText = String(seconds).padStart(2, "0");

    // If the countdown is over, display a message
    if (timeRemaining < 0) {
        clearInterval(countdownTimer);
        document.querySelector(".timer-container").innerHTML = "EXPIRED";
    }
}, 1000);


// Get elements
const supportBtn = document.getElementById('supportBtn');
const chatbotIframeContainer = document.getElementById('chatbotIframeContainer');
const closeChatbot = document.getElementById('closeChatbot');

// Open chatbot iframe
supportBtn.addEventListener('click', () => {
    chatbotIframeContainer.style.display = 'flex'; /* Use flex to align the header and iframe */
});

// Close chatbot iframe
closeChatbot.addEventListener('click', () => {
    chatbotIframeContainer.style.display = 'none';
});

// Close iframe when clicking outside
window.addEventListener('click', (event) => {
    if (event.target === chatbotIframeContainer) {
        chatbotIframeContainer.style.display = 'none';
    }
});

// Chat Widget Implementation
document.addEventListener('DOMContentLoaded', function() {
    // Get chat elements
    const chatWidget = document.getElementById('chat-widget');
    const chatToggle = document.getElementById('chat-toggle');
    const chatContainer = document.getElementById('chat-container');
    const closeChat = document.getElementById('close-chat');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-message');
    const chatMessages = document.getElementById('chat-messages');

    if (!chatWidget || !chatToggle || !chatContainer) {
        console.error('Chat elements not found!');
        return;
    }

    // Initialize chat
    let isFirstOpen = true;

    // Toggle chat window
    chatToggle.addEventListener('click', () => {
        chatContainer.classList.toggle('hidden');
        if (!chatContainer.classList.contains('hidden')) {
            userInput.focus();
            if (isFirstOpen) {
                addBotMessage("👋 Hello! I'm your AI assistant. How can I help you today?");
                isFirstOpen = false;
            }
        }
    });

    // Close chat
    closeChat.addEventListener('click', () => {
        chatContainer.classList.add('hidden');
    });

    // Send message on button click
    sendButton.addEventListener('click', () => sendMessage());

    // Send message on Enter key
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            sendMessage();
        }
    });

    function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;

        // Add user message
        addUserMessage(message);
        userInput.value = '';

        // Show typing indicator
        const typingIndicator = addTypingIndicator();

        // Send to backend
        fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                user_input: message,
                session_id: Date.now().toString() 
            })
        })
        .then(response => response.json())
        .then(data => {
            typingIndicator.remove();
            addBotResponse(data);
        })
        .catch(error => {
            console.error('Error:', error);
            typingIndicator.remove();
            addBotMessage('Sorry, I encountered an error. Please try again.');
        });
    }

    // Helper functions
    function addUserMessage(text) {
        const message = document.createElement('div');
        message.className = 'message user-message';
        message.textContent = text;
        chatMessages.appendChild(message);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function addBotMessage(text) {
        const message = document.createElement('div');
        message.className = 'message bot-message';
        message.textContent = text;
        chatMessages.appendChild(message);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function addBotResponse(data) {
        const response = document.createElement('div');
        response.className = 'message bot-message';

        // Add main message
        if (data.message) {
            const messageText = document.createElement('div');
            messageText.textContent = data.message;
            response.appendChild(messageText);
        }

        // Add products if present
        if (data.response && data.response.products) {
            const productsContainer = document.createElement('div');
            productsContainer.className = 'products-container';
            
            data.response.products.forEach(product => {
                productsContainer.innerHTML += `
                    <div class="product-card">
                        <h4>${product.name}</h4>
                        <p>${product.description}</p>
                        <div class="price">$${product.price}</div>
                        ${product.rating ? `<div class="rating">★ ${product.rating}/5</div>` : ''}
                    </div>
                `;
            });
            response.appendChild(productsContainer);
        }

        // Add suggestions if present
        if (data.response && data.response.suggestions) {
            const suggestions = document.createElement('div');
            suggestions.className = 'suggestion-chips';
            data.response.suggestions.forEach(suggestion => {
                const chip = document.createElement('span');
                chip.className = 'suggestion-chip';
                chip.textContent = suggestion;
                chip.onclick = () => {
                    userInput.value = suggestion;
                    sendMessage();
                };
                suggestions.appendChild(chip);
            });
            response.appendChild(suggestions);
        }

        chatMessages.appendChild(response);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function addTypingIndicator() {
        const indicator = document.createElement('div');
        indicator.className = 'typing-indicator';
        indicator.innerHTML = '<span></span><span></span><span></span>';
        chatMessages.appendChild(indicator);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return indicator;
    }
});