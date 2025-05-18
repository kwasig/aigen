import AddProviderForm from '@/components/settings/add-provider-form';

export default function ProvidersSettingsPage() {
  return (
    <div className="space-y-6">
      <h2 className="text-xl font-semibold">Add LLM Provider</h2>
      <AddProviderForm />
    </div>
  );
}
