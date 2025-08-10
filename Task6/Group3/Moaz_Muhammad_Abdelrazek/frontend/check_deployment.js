#!/usr/bin/env node
/**
 * Frontend Deployment Validation Script for Render Free Tier
 */

const fs = require('fs');
const path = require('path');

function checkFrontendReady() {
    console.log('🔍 Checking Frontend Deployment Readiness...');
    
    const issues = [];
    
    // Check required files
    const requiredFiles = [
        'package.json',
        'src/App.tsx',
        'src/services/api.ts',
        'public/index.html'
    ];
    
    for (const file of requiredFiles) {
        if (!fs.existsSync(file)) {
            issues.push(`❌ Missing file: ${file}`);
        } else {
            console.log(`✅ Found: ${file}`);
        }
    }
    
    // Check package.json
    try {
        const packageJson = JSON.parse(fs.readFileSync('package.json', 'utf8'));
        
        if (!packageJson.scripts.build) {
            issues.push('❌ Missing build script in package.json');
        }
        
        if (!packageJson.dependencies.react) {
            issues.push('❌ React not found in dependencies');
        }
        
        if (!packageJson.dependencies.axios) {
            issues.push('❌ Axios not found in dependencies');
        }
        
        console.log(`📦 React version: ${packageJson.dependencies.react || 'not found'}`);
        
    } catch (e) {
        issues.push(`❌ Error reading package.json: ${e.message}`);
    }
    
    // Check if API service is configured
    try {
        const apiContent = fs.readFileSync('src/services/api.ts', 'utf8');
        if (!apiContent.includes('REACT_APP_API_URL')) {
            issues.push('❌ API URL environment variable not configured');
        } else {
            console.log('✅ API URL configuration found');
        }
    } catch (e) {
        issues.push(`❌ Error reading api.ts: ${e.message}`);
    }
    
    // Summary
    if (issues.length > 0) {
        console.log('\n⚠️ Issues found:');
        issues.forEach(issue => console.log(`  ${issue}`));
        console.log('\n🔧 Please fix these issues before deploying.');
        return false;
    } else {
        console.log('\n✅ Frontend is ready for Render deployment!');
        console.log('\n📋 Deployment Commands:');
        console.log('Build Command: npm ci && npm run build');
        console.log('Publish Directory: build');
        console.log('\n⚠️ Don\'t forget to set REACT_APP_API_URL environment variable!');
        return true;
    }
}

const success = checkFrontendReady();
process.exit(success ? 0 : 1);
